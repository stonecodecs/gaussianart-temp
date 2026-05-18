#!/usr/bin/env python3

"""
TSDF fusion + mesh extraction for GaussianArt outputs.

Supports:
- full object reconstruction
- static-only reconstruction  (requires --partnet_root)
- dynamic-only reconstruction (requires --partnet_root)
- Open3D scalable TSDF fusion

usage:
python tsdf_meshify.py --object_root ./output/table/Table_34610 \\
    --partnet_root ./partnet-mobility-video-dataset

Expected directory structure:
    object_root/
        cameras.json          # GaussianArt format: flat list of camera dicts
        meshes/               # output directory
        train/ours_<N>/renders/   # default RGB source (render.py train views)
            0000.png ...        # matched to cameras.json by img_name
        test/ours_<N>/renders/    # use --split test
        video/ours_<N>/renders/   # orbital demo; use --split video

    partnet_root/             # required for metric depth + semantic masks
        <object_name>/
            multiview_static/<split>/
                depth/        # float32 .npy metric depth in metres
                semantic/     # int32  .npy part labels (-1=bg, 0=static, 1+=dynamic)

GaussianArt cameras.json format:
    [
      {
        "id": 0,
        "img_name": "0000",   # 4-digit name matching PartNet files
        "width": 640,
        "height": 480,
        "fx": ..., "fy": ...,
        "position": [x, y, z],          # camera centre in world (c2w translation)
        "rotation": [[...], [...], [...]] # 3x3 c2w rotation (OpenCV convention)
      },
      ...
    ]

Requirements:
pip install open3d opencv-python tqdm numpy

Outputs:
    meshes/
        tsdf_mesh_full.ply
        tsdf_mesh_static.ply
        tsdf_mesh_dynamic.ply
"""

import os
import json
import glob
import argparse

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm


# ============================================================
# CONFIG DEFAULTS
# ============================================================

DEFAULT_VOXEL_SIZE = 0.003
DEFAULT_SDF_TRUNC = 0.015
DEFAULT_DEPTH_MAX = 6.0

# semantic label conventions (PartNet-Mobility)
STATIC_LABEL = 0   # label 0 = static base; labels >=1 are dynamic parts
BACKGROUND_LABEL = -1


# ============================================================
# UTILITIES
# ============================================================

def find_latest_ours_subdir(parent_dir: str) -> str:
    """Return the basename of the highest-numbered ``ours_*`` under *parent_dir*."""
    candidates = sorted(glob.glob(os.path.join(parent_dir, "ours_*")))
    if not candidates:
        raise RuntimeError(f"No ours_* subdirectory found in {parent_dir}")
    return os.path.basename(candidates[-1])


def resolve_render_dir(object_root: str, split: str, render_subdir: str | None) -> tuple[str, str]:
    """
    Resolve the directory of RGB renders and the ``ours_<N>`` subdir name.

    *split* is ``train``, ``test``, or ``video`` (orbital frames under ``video/``).
    """
    if split not in ("train", "test", "video"):
        raise ValueError(f"split must be train, test, or video; got {split!r}")
    parent = os.path.join(object_root, split)
    if not os.path.isdir(parent):
        raise RuntimeError(f"Render parent directory not found: {parent}")
    if render_subdir is None:
        render_subdir = find_latest_ours_subdir(parent)
    render_dir = os.path.join(parent, render_subdir, "renders")
    if not os.path.isdir(render_dir):
        raise RuntimeError(f"Render directory not found: {render_dir}")
    return render_dir, render_subdir


def find_render_frames(render_dir):
    """
    Collect render PNGs named with digits only (e.g. 0000.png or 00000.png).
    Files with an underscore (e.g. 0_00000.png) are skipped.
    """
    all_pngs = sorted(glob.glob(os.path.join(render_dir, "*.png")))
    return [p for p in all_pngs if "_" not in os.path.basename(p)]


def build_camera_lookup(cameras_list: list) -> dict[str, int]:
    """
    Map ``img_name`` -> index in *cameras_list*.

    Duplicate names can occur (start + end multiview); the last entry wins so
    end-state cameras take precedence when both are present.
    """
    by_name: dict[str, int] = {}
    for i, frame in enumerate(cameras_list):
        by_name[frame["img_name"]] = i
    return by_name


def resolve_img_name(frame_stem: str, camera_by_img: dict[str, int]) -> str | None:
    """Match a render filename stem to a cameras.json ``img_name``."""
    candidates = []
    for c in (frame_stem, frame_stem.zfill(4), frame_stem.lstrip("0") or "0"):
        if c not in candidates:
            candidates.append(c)
    for name in candidates:
        if name in camera_by_img:
            return name
    return None


def load_camera_json(camera_json_path):
    """
    Load GaussianArt cameras.json, which is a flat JSON array.
    """
    with open(camera_json_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError(
            f"Expected cameras.json to be a JSON array, got {type(data).__name__}. "
            "If your file uses the NeRF {'frames': [...]} format, pass "
            "--camera_format nerf."
        )
    return data


def get_frame_camera(cameras_list, list_idx):
    """
    Build Open3D intrinsic + extrinsic from a GaussianArt camera entry.

    GaussianArt convention:
        rotation  – 3×3 camera-to-world rotation (OpenCV: camera +Z forward)
        position  – camera centre in world coordinates (c2w translation)

    Open3D expects the w2c (world-to-camera) matrix as the extrinsic,
    so we build c2w = [[R | t]; [0 0 0 1]] and invert it.
    """
    frame = cameras_list[list_idx]

    fx = frame["fx"]
    fy = frame["fy"]
    width  = frame["width"]
    height = frame["height"]

    # cx/cy are not stored; derive from image dimensions (principal point at centre)
    cx = (width  - 1) / 2.0
    cy = (height - 1) / 2.0

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )

    R = np.array(frame["rotation"], dtype=np.float64)   # 3×3
    t = np.array(frame["position"], dtype=np.float64)   # (3,)

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R
    c2w[:3,  3] = t

    extrinsic = np.linalg.inv(c2w)   # w2c for Open3D

    return intrinsic, extrinsic


def load_partnet_depth(partnet_split_dir, img_name):
    """
    Load metric depth (float32, metres) from a PartNet-Mobility split directory.

    Args:
        partnet_split_dir: path to e.g. .../multiview_static/train
        img_name: 4-digit string, e.g. '0023'

    Returns:
        depth: float32 ndarray (H, W), background pixels set to 0
    """
    path = os.path.join(partnet_split_dir, "depth", f"{img_name}.npy")
    if not os.path.exists(path):
        return None
    depth = np.load(path).astype(np.float32)
    # PartNet uses 1e10 as the background sentinel
    depth[depth > 1e9] = 0.0
    return depth


def load_partnet_semantic(partnet_split_dir, img_name):
    """
    Load semantic part label map (int32) from PartNet-Mobility.

    Labels:
        -1  background
         0  static base
        ≥1  dynamic parts
    """
    path = os.path.join(partnet_split_dir, "semantic", f"{img_name}.npy")
    if not os.path.exists(path):
        return None
    return np.load(path).astype(np.int32)


# ============================================================
# TSDF
# ============================================================

def create_tsdf_volume(voxel_size, sdf_trunc):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    return volume


def integrate_frame(volume, rgb, depth, intrinsic, extrinsic, depth_max):
    """
    Integrate one RGBD frame into the TSDF volume.

    depth must already be in metres (float32).  We pass depth_scale=1.0
    so Open3D reads the values directly.
    """
    color_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
    depth_o3d = o3d.geometry.Image(depth.astype(np.float32))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0,
        depth_trunc=depth_max,
        convert_rgb_to_intensity=False,
    )

    volume.integrate(rgbd, intrinsic, extrinsic)


# ============================================================
# MAIN
# ============================================================

def main(args):

    object_name = os.path.basename(os.path.normpath(args.object_root))

    # ------------------------------------------------------------------
    # Resolve RGB render directory (train / test / video)
    # ------------------------------------------------------------------
    render_subdir = args.render_subdir or args.video_subdir
    render_dir, ours_name = resolve_render_dir(
        args.object_root, args.split, render_subdir
    )
    print(f"RGB renders: {args.split}/{ours_name}/renders/ ({render_dir})")

    mesh_dir = os.path.join(args.object_root, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # PartNet split directory (for metric depth + semantics)
    # ------------------------------------------------------------------
    partnet_split_name = args.partnet_split
    if partnet_split_name is None:
        partnet_split_name = "test" if args.split == "test" else "train"

    partnet_split = None
    if args.partnet_root is not None:
        partnet_split = os.path.join(
            args.partnet_root,
            object_name,
            args.partnet_multiview,
            partnet_split_name,
        )
        if not os.path.isdir(partnet_split):
            raise RuntimeError(
                f"PartNet split directory not found: {partnet_split}\n"
                "Expected layout: <partnet_root>/<object_name>/multiview_static/train/"
            )

    if args.mode != "full" and partnet_split is None:
        raise RuntimeError(
            "--partnet_root is required for static/dynamic modes "
            "(needed for metric depth and semantic masks)"
        )

    # ------------------------------------------------------------------
    # Cameras
    # ------------------------------------------------------------------
    camera_json_path = os.path.join(args.object_root, "cameras.json")
    cameras_list = load_camera_json(camera_json_path)
    camera_by_img = build_camera_lookup(cameras_list)

    # ------------------------------------------------------------------
    # Render frames
    # ------------------------------------------------------------------
    render_paths = find_render_frames(render_dir)
    if len(render_paths) == 0:
        raise RuntimeError(
            f"No valid render frames found in {render_dir}.\n"
            "Frames must be named with only digits (e.g. 0000.png); "
            "files with underscores are skipped."
        )
    print(f"Found {len(render_paths)} render frames")

    # ------------------------------------------------------------------
    # TSDF volume
    # ------------------------------------------------------------------
    volume = create_tsdf_volume(voxel_size=args.voxel_size, sdf_trunc=args.sdf_trunc)

    skipped = 0

    for render_path in tqdm(render_paths):

        frame_stem = os.path.splitext(os.path.basename(render_path))[0]
        img_name = resolve_img_name(frame_stem, camera_by_img)
        if img_name is None:
            print(f"  [warn] no camera for render {frame_stem} in cameras.json — skipping")
            skipped += 1
            continue

        list_idx = camera_by_img[img_name]

        # ------------------------------------------------------------------
        # Colour image
        # ------------------------------------------------------------------
        rgb = cv2.imread(render_path)
        if rgb is None:
            print(f"  [warn] could not read {render_path} — skipping")
            skipped += 1
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # ------------------------------------------------------------------
        # Depth (metric, metres)
        # ------------------------------------------------------------------
        depth = None

        if partnet_split is not None:
            depth = load_partnet_depth(partnet_split, img_name)

        if depth is None:
            print(f"  [warn] no metric depth for frame {frame_stem} "
                  f"(img_name={img_name}) — skipping")
            skipped += 1
            continue

        # ------------------------------------------------------------------
        # Semantic mask → zero-out depth outside the region of interest
        # ------------------------------------------------------------------
        if args.mode != "full":
            sem = load_partnet_semantic(partnet_split, img_name)
            if sem is None:
                print(f"  [warn] no semantic for frame {frame_stem} — skipping")
                skipped += 1
                continue

            if args.mode == "static":
                keep = (sem == STATIC_LABEL)
            else:  # dynamic
                keep = (sem != STATIC_LABEL) & (sem != BACKGROUND_LABEL)

            depth[~keep] = 0.0

        # ------------------------------------------------------------------
        # Depth sanity cleanup
        # ------------------------------------------------------------------
        depth[~np.isfinite(depth)] = 0.0
        depth[depth < args.min_depth] = 0.0
        depth[depth > args.depth_max] = 0.0

        # ------------------------------------------------------------------
        # Build camera matrices
        # ------------------------------------------------------------------
        intrinsic, extrinsic = get_frame_camera(cameras_list, list_idx)

        # ------------------------------------------------------------------
        # Integrate
        # ------------------------------------------------------------------
        integrate_frame(
            volume=volume,
            rgb=rgb,
            depth=depth,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            depth_max=args.depth_max,
        )

    print(f"Integration complete ({len(render_paths) - skipped} frames used, "
          f"{skipped} skipped)")

    # ======================================================================
    # Mesh extraction
    # ======================================================================
    print("Extracting mesh...")

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()

    # ======================================================================
    # Save
    # ======================================================================
    output_name = f"tsdf_mesh_{args.mode}.ply"
    output_path = os.path.join(mesh_dir, output_name)

    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"Saved mesh to: {output_path}")
    print(mesh)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="TSDF fusion for GaussianArt outputs using PartNet-Mobility GT depth."
    )

    parser.add_argument(
        "--object_root",
        type=str,
        required=True,
        help="Path to GaussianArt output directory for one object "
             "(cameras.json plus train|test|video/ours_<N>/renders/).",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "video"],
        help="Which render folder to fuse (default: train views from render.py).",
    )

    parser.add_argument(
        "--render_subdir",
        type=str,
        default=None,
        help="ours_<N> subdirectory (e.g. ours_65000). Default: latest under "
             "<split>/.",
    )

    parser.add_argument(
        "--video_subdir",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--partnet_root",
        type=str,
        default=None,
        help="Root of the PartNet-Mobility video dataset "
             "(e.g. .../partnet-mobility-video-dataset). "
             "Required for all modes; provides metric depth (.npy) and "
             "semantic part masks (.npy).",
    )

    parser.add_argument(
        "--partnet_multiview",
        type=str,
        default="multiview_static",
        help="Multiview folder under each object in partnet_root "
             "(default: multiview_static / end state).",
    )

    parser.add_argument(
        "--partnet_split",
        type=str,
        default=None,
        choices=["train", "test"],
        help="PartNet train/ or test/ subfolder for depth and semantics "
             "(default: train, or test when --split test).",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "static", "dynamic"],
        help="Which part of the object to reconstruct. "
             "'static' and 'dynamic' require --partnet_root.",
    )

    # ------------------------------------------------------------------
    # TSDF hyper-parameters
    # ------------------------------------------------------------------

    parser.add_argument(
        "--voxel_size",
        type=float,
        default=DEFAULT_VOXEL_SIZE,
        help="TSDF voxel size in metres (default %(default)s).",
    )

    parser.add_argument(
        "--sdf_trunc",
        type=float,
        default=DEFAULT_SDF_TRUNC,
        help="SDF truncation distance in metres (default %(default)s).",
    )

    parser.add_argument(
        "--depth_max",
        type=float,
        default=DEFAULT_DEPTH_MAX,
        help="Maximum depth in metres; deeper pixels are ignored (default %(default)s).",
    )

    parser.add_argument(
        "--min_depth",
        type=float,
        default=0.05,
        help="Minimum depth in metres; shallower pixels are ignored (default %(default)s).",
    )

    args = parser.parse_args()

    main(args)