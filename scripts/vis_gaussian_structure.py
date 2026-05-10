#!/usr/bin/env python3
"""
Visualize 3D Gaussian Splatting primitives as explicit ellipsoid meshes.

Supports:
  • Interactive Open3D viewer (single PLY)
  • MP4 video with the same orbital camera trajectory as render_video.py
  • MP4 video following cameras listed in a trained scene's cameras.json

Expected PLY fields (GaussianArt / 3DGS):
    x y z, scale_0..2 (log), rot_0..3 (quaternion w,x,y,z)

Install:
    pip install open3d plyfile numpy opencv-python

Examples:
    # Interactive (subsampled ellipsoids)
    python scripts/vis_gaussian_structure.py --ply path/to/point_cloud.ply

    # Orbital video (smooth motion from frame 0; defaultLit + sun), 30 fps
    python scripts/vis_gaussian_structure.py -m output/MPArt90/Table_34610 --video \\
        --trajectory orbit --out ellipsoid_orbit.mp4

    # Gaussian DC colors as ellipsoid albedo (needs f_dc_* in PLY)
    python scripts/vis_gaussian_structure.py -m output/MPArt90/Table_34610 --video \\
        --use_gaussian_color --out ellipsoid_colored.mp4

    # Legacy orbit hold for the first half of frames (same math as render_video.py)
    python scripts/vis_gaussian_structure.py -m output/MPArt90/Table_34610 --video \\
        --orbit_static_intro --out ellipsoid_hold.mp4

    # Video along training camera path from cameras.json
    python scripts/vis_gaussian_structure.py -m output/MPArt90/Table_34610 --video \\
        --trajectory cameras_json --out ellipsoid_cams.mp4

Camera / coordinates:
    Video uses Open3D ``setup_camera(fov, center, eye, up)`` (Filament look-at),
    not raw OpenCV world→camera matrices. ``cameras.json`` stores COLMAP-style
    extrinsics; passing those matrices with ``PinholeCameraIntrinsic`` produced
    blank frames. Orbital poses interpreted as OpenCV W2C also pointed away from
    the scene (negative Z). Each frame uses the correct camera **world position**
    and looks at the mesh bounding-box center with world up ≈ +Z.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
from plyfile import PlyData


# ---------------------------------------------------------------------------
# Quaternion / ellipsoid (same convention as typical 3DGS PLY)
# ---------------------------------------------------------------------------

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """q: [w, x, y, z] normalized."""
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def create_gaussian_ellipsoid(
    center: np.ndarray,
    scales: np.ndarray,
    rotation: np.ndarray,
    color=(0.65, 0.65, 0.65),
    resolution: int = 8,
) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
    vertices = np.asarray(mesh.vertices)
    S = np.diag(scales.astype(np.float64))
    R = quaternion_to_rotation_matrix(rotation.astype(np.float64))
    vertices = (R @ (S @ vertices.T)).T + center
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(list(color))
    return mesh


# SH DC → RGB (same as utils.sh_utils.SH2RGB for degree-0 coeffs)
_SH_C0 = 0.28209479177387814


def load_gaussians(ply_path: str):
    ply = PlyData.read(ply_path)
    vertex = ply["vertex"]
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    scales = np.exp(
        np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1)
    )
    rotations = np.stack(
        [vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=1
    )
    return xyz, scales, rotations


def load_gaussian_dc_rgb(ply_path: str) -> np.ndarray | None:
    """Per-Gaussian RGB from PLY ``f_dc_*`` (view-independent DC color)."""
    ply = PlyData.read(ply_path)
    vertex = ply["vertex"]
    props = {p.name for p in vertex.properties}
    if not {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(props):
        return None
    sh = np.stack(
        [vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1
    ).astype(np.float64)
    rgb = sh * _SH_C0 + 0.5
    return np.clip(rgb, 0.0, 1.0)


def build_combined_ellipsoid_mesh(
    xyz: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    max_gaussians: int,
    sphere_resolution: int,
    scale_multiplier: float,
    color=(0.58, 0.58, 0.60),
    seed: int = 0,
    per_gaussian_rgb: np.ndarray | None = None,
) -> o3d.geometry.TriangleMesh:
    n = len(xyz)
    if max_gaussians > 0 and n > max_gaussians:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, max_gaussians, replace=False)
        xyz, scales, rotations = xyz[idx], scales[idx], rotations[idx]
        if per_gaussian_rgb is not None:
            per_gaussian_rgb = per_gaussian_rgb[idx]

    combined = o3d.geometry.TriangleMesh()
    for i, (center, scale, rot) in enumerate(zip(xyz, scales, rotations)):
        if per_gaussian_rgb is not None:
            c = tuple(np.asarray(per_gaussian_rgb[i], dtype=np.float64).flatten()[:3])
        else:
            c = color
        combined += create_gaussian_ellipsoid(
            center=np.asarray(center, dtype=np.float64),
            scales=np.asarray(scale, dtype=np.float64) * scale_multiplier,
            rotation=np.asarray(rot, dtype=np.float64),
            color=c,
            resolution=sphere_resolution,
        )
    combined.compute_vertex_normals()
    return combined


# ---------------------------------------------------------------------------
# Camera trajectories (orbit = same math as render_video.py)
# ---------------------------------------------------------------------------

def generate_camera_poses(N: int = 30, static_intro: bool = False, loopable: bool = True) -> list[np.ndarray]:
    """
    Spherical orbit matching ``render_video.generate_camera_poses`` parameters.

    By default (``static_intro=False``) theta and phi both sweep smoothly over all
    ``N`` frames so the camera moves immediately. The original implementation
    prepended ``N//2`` identical samples (fixed theta & phi), which freezes motion
    for the first half of the trajectory — set ``static_intro=True`` to reproduce
    that legacy behavior.
    
    Args:
        N: Number of frames
        static_intro: Legacy behavior with static first half
        loopable: If True, ensure first and last frames connect smoothly (default True)
    """
    poses = []
    traj_info = {
        "radius": 3.5,
        "theta": [-0, 0.1],
        "d_theta": 0.2,
        "phi": [0, 2],
        "d_phi": -0.75,
        "rotx90": 0,
        "roty180": 0,
    }
    radius, r_theta, r_phi = traj_info["radius"], traj_info["theta"], traj_info["phi"]
    d_theta, d_phi = traj_info["d_theta"], traj_info["d_phi"]

    if static_intro:
        # Legacy render_video.py prepended N//2 duplicate samples; match that hold
        # while emitting exactly N poses (concat in the old code produced N+N//2).
        half = N // 2
        rest = max(N - half, 1)
        theta_fixed = d_theta * np.pi
        phi_fixed = d_phi * np.pi
        theta_sweep = (
            np.linspace(r_theta[0] * np.pi, r_theta[1] * np.pi, rest) + d_theta * np.pi
        )
        phi_sweep = (
            np.linspace(r_phi[0] * np.pi, r_phi[1] * np.pi, rest) + d_phi * np.pi
        )
        thetas = np.concatenate([np.full(half, theta_fixed), theta_sweep])
        azimuths = np.concatenate([np.full(half, phi_fixed), phi_sweep])
    else:
        # For loopable: use endpoint=False so last frame connects to first
        # For non-loopable: use endpoint=True (standard behavior)
        thetas = np.linspace(r_theta[0] * np.pi, r_theta[1] * np.pi, N, endpoint=not loopable) + d_theta * np.pi
        azimuths = np.linspace(r_phi[0] * np.pi, r_phi[1] * np.pi, N, endpoint=not loopable) + d_phi * np.pi

    roty180 = (
        np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        if traj_info["roty180"]
        else np.eye(4)
    )
    rotx90 = (
        np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        if traj_info["rotx90"]
        else np.eye(4)
    )

    for theta, azimuth in zip(thetas, azimuths):
        x = radius * np.cos(azimuth) * np.cos(theta)
        y = radius * np.sin(azimuth) * np.cos(theta)
        z = radius * np.sin(theta)
        position = np.array([x, y, z])
        forward = position / (np.linalg.norm(position) + 1e-12)
        up = np.array([0.0, 0.0, 1.0])
        if np.allclose(forward, up) or np.allclose(forward, -up):
            up = np.array([0.0, 1.0, 0.0])
        right = np.cross(up, forward)
        up = np.cross(forward, right)
        right /= np.linalg.norm(right) + 1e-12
        up /= np.linalg.norm(up) + 1e-12
        rotation_matrix = np.vstack([right, up, forward]).T
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = position
        transformation_matrix = roty180 @ rotx90.T @ transformation_matrix
        poses.append(transformation_matrix)
    return poses


def camera_center_from_opencv_w2c(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Camera center C_w given OpenCV-style world→camera: P_c = R @ P_w + t."""
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    return (-R.T @ t.reshape(3, 1)).flatten()


def vertical_fov_y_deg(height: int, fy: float) -> float:
    """Full vertical field of view in degrees (matches fy from cameras.json)."""
    fy = float(fy)
    return float(np.degrees(2.0 * np.arctan(height / (2.0 * fy))))


def world_up_for_look_at(
    eye: np.ndarray, center: np.ndarray, preferred=(0.0, 0.0, 1.0)
) -> np.ndarray:
    """World-space up vector for Open3D look-at; avoids parallel view axis."""
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    center = np.asarray(center, dtype=np.float64).reshape(3)
    up = np.asarray(preferred, dtype=np.float64).reshape(3)
    view = center - eye
    view /= np.linalg.norm(view) + 1e-12
    if abs(np.dot(view, up)) > 0.92:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return up.astype(np.float32).reshape(3, 1)


def load_cameras_json(path: str) -> tuple[list[np.ndarray], list[tuple[int, int, float, float]]]:
    """
    Load cameras.json written by utils.camera_utils.camera_to_JSON.

    Returns:
        w2c_list: list of 4x4 world-to-camera matrices
        intrinsics: parallel list of (width, height, fx, fy)
    """
    with open(path, "r") as f:
        cams = json.load(f)
    cams = sorted(cams, key=lambda c: c["id"])
    w2c_list = []
    intrinsics = []
    for cam in cams:
        W2C = np.eye(4, dtype=np.float64)
        W2C[:3, :3] = np.array(cam["rotation"], dtype=np.float64)
        W2C[:3, 3] = np.array(cam["position"], dtype=np.float64)
        w2c_list.append(W2C)
        intrinsics.append(
            (int(cam["width"]), int(cam["height"]), float(cam["fx"]), float(cam["fy"]))
        )
    return w2c_list, intrinsics


def default_intrinsic_from_cameras_json(
    cam_json_path: Path, 
    override_width: int | None = None,
    override_height: int | None = None
) -> tuple[int, int, float, float]:
    """
    Get intrinsics from cameras.json with optional resolution override.
    
    If width/height are overridden, focal lengths are scaled proportionally.
    """
    if not cam_json_path.is_file():
        raise FileNotFoundError(f"Missing {cam_json_path}")
    _, intr = load_cameras_json(str(cam_json_path))
    w, h, fx, fy = intr[0]
    
    # Scale focal lengths if resolution is overridden
    if override_width is not None:
        scale_x = override_width / w
        fx = fx * scale_x
        w = override_width
    
    if override_height is not None:
        scale_y = override_height / h
        fy = fy * scale_y
        h = override_height
    
    return w, h, fx, fy


# ---------------------------------------------------------------------------
# Offscreen rendering → MP4
# ---------------------------------------------------------------------------

def render_frames_to_video(
    mesh: o3d.geometry.TriangleMesh,
    frames_eye_world: list[np.ndarray],
    look_at_center: np.ndarray,
    frames_intrinsic: list[tuple[int, int, float, float]],
    output_path: str,
    fps: float = 30.0,
    bg_color=(1.0, 1.0, 1.0, 1.0),
    bbox_diagonal: float = 1.0,
    lit: bool = True,
    base_gray: tuple[float, float, float] = (0.58, 0.58, 0.60),
    use_vertex_albedo: bool = False,
) -> None:
    """
    Render using Open3D's OpenGL-style look-at API (FOV + center + eye + up).

    PinholeCameraIntrinsic + extrinsic matrices are brittle here because 3DGS /
    COLMAP use an OpenCV-style world→camera convention (+Z forward in camera
    space) while Filament expects a classic GL view setup; passing raw W2C
    matrices produced blank frames. Orbital poses from ``generate_camera_poses``
    also pointed **away** from the scene origin when interpreted as OpenCV W2C,
    leaving everything behind the camera (negative Z).
    """
    if len(frames_eye_world) != len(frames_intrinsic):
        raise ValueError("frames_eye_world and frames_intrinsic length mismatch")

    w0, h0 = frames_intrinsic[0][0], frames_intrinsic[0][1]
    for fw, fh, _, _ in frames_intrinsic:
        if (fw, fh) != (w0, h0):
            raise RuntimeError(
                f"Varying image sizes in trajectory {(w0,h0)} vs {(fw,fh)}; "
                "resize data or use orbit mode."
            )

    center_col = np.asarray(look_at_center, dtype=np.float32).reshape(3, 1)
    diag = max(float(bbox_diagonal), 1e-6)

    renderer = rendering.OffscreenRenderer(w0, h0)
    renderer.scene.set_background(bg_color)

    mat = rendering.MaterialRecord()
    scene_ = renderer.scene.scene
    if lit:
        mat.shader = "defaultLit"
        if use_vertex_albedo:
            # Ellipsoids carry per-vertex albedo from Gaussian DC color.
            mat.base_color = (1.0, 1.0, 1.0, 1.0)
        else:
            mat.base_color = (base_gray[0], base_gray[1], base_gray[2], 1.0)
        mat.base_metallic = 0.04
        mat.base_roughness = 0.52
        mat.base_reflectance = 0.35
        try:
            scene_.enable_sun_light(True)
            sun_dir = np.array([[0.38], [0.22], [0.90]], dtype=np.float32)
            sun_dir /= np.linalg.norm(sun_dir)
            scene_.set_sun_light(
                sun_dir,
                np.array([[1.0], [0.98], [0.95]], dtype=np.float32),
                105000.0,
            )
        except Exception:
            pass
        try:
            scene_.enable_indirect_light(True)
            scene_.set_indirect_light_intensity(35000.0)
        except Exception:
            pass
    else:
        mat.shader = "defaultUnlit"
        mat.base_color = (base_gray[0], base_gray[1], base_gray[2], 1.0)

    renderer.scene.add_geometry("gaussians", mesh, mat)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_path, fourcc, fps, (w0, h0))

    near_clip = max(0.01, diag * 5e-4)
    far_clip = diag * 80.0

    for eye_world, (fw, fh, _fx, ffy) in zip(frames_eye_world, frames_intrinsic):
        eye_col = np.asarray(eye_world, dtype=np.float32).reshape(3, 1)
        up_col = world_up_for_look_at(eye_col.flatten(), center_col.flatten())
        fov_y_deg = vertical_fov_y_deg(fh, ffy)

        # Open3D order: vertical_fov_deg, center (look-at), eye (camera position), up
        renderer.setup_camera(
            fov_y_deg,
            center_col,
            eye_col,
            up_col,
            near_clip,
            far_clip,
        )
        img = renderer.render_to_image()
        rgb = np.asarray(img)[:, :, :3].astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        video.write(bgr)

    video.release()
    try:
        renderer.scene.remove_geometry("gaussians")
    except Exception:
        pass
    print(f"Wrote video: {output_path}")


def resolve_ply(model_dir: Path, ply_arg: str | None, iteration: int | None) -> Path:
    if ply_arg:
        p = Path(ply_arg)
        if not p.is_file():
            raise FileNotFoundError(ply_arg)
        return p.resolve()

    pc_root = model_dir / "point_cloud"
    if not pc_root.is_dir():
        raise FileNotFoundError(f"No point_cloud/ under {model_dir}")

    if iteration is not None:
        cand = pc_root / f"iteration_{iteration}" / "point_cloud.ply"
        if not cand.is_file():
            raise FileNotFoundError(cand)
        return cand

    it_dirs = sorted(glob.glob(str(pc_root / "iteration_*")))
    if not it_dirs:
        raise FileNotFoundError(f"No iteration_* under {pc_root}")
    latest = Path(it_dirs[-1])
    ply = latest / "point_cloud.ply"
    if not ply.is_file():
        raise FileNotFoundError(ply)
    return ply.resolve()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize 3DGS Gaussians as ellipsoids (viewer or video)."
    )
    parser.add_argument("--ply", type=str, default=None, help="Path to point_cloud.ply")
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default=None,
        help="Trained output dir (e.g. output/MPArt90/Table_34610); finds latest PLY + cameras.json",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Use point_cloud/iteration_<n>/; default: latest",
    )
    parser.add_argument(
        "--max_gaussians",
        type=int,
        default=1_000_000,
        help="Cap ellipsoid count (subsample randomly). Use 0 for no cap / all Gaussians.",
    )
    parser.add_argument("--sphere_resolution", type=int, default=8)
    parser.add_argument("--scale_multiplier", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--video",
        action="store_true",
        help="Render MP4 instead of interactive viewer",
    )
    parser.add_argument(
        "--trajectory",
        choices=("orbit", "cameras_json"),
        default="orbit",
        help="orbit: same path as render_video.py; cameras_json: use cameras.json poses",
    )
    parser.add_argument(
        "--cameras_json",
        type=str,
        default=None,
        help="Override path to cameras.json (default: <model_path>/cameras.json)",
    )
    parser.add_argument("--n_frames", type=int, default=90, help="Frames for orbit mode")
    parser.add_argument(
        "--max_camera_frames",
        type=int,
        default=None,
        help="Use only first N poses from cameras_json (debug / shorter video)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Video frame rate (default 30)",
    )
    parser.add_argument(
        "--orbit_static_intro",
        action="store_true",
        help="Legacy orbit: first half of frames hold fixed theta/phi (matches old render_video feel)",
    )
    parser.add_argument(
        "--no_loop",
        action="store_true",
        help="Disable loopable orbit (end frame won't connect to start)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Override output video width (scales fx proportionally)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Override output video height (scales fy proportionally)",
    )
    parser.add_argument(
        "--use_gaussian_color",
        action="store_true",
        help="Color each ellipsoid from PLY f_dc_0..2 (SH DC, same as 3DGS base albedo)",
    )
    parser.add_argument(
        "--unlit",
        action="store_true",
        help="Disable sun/indirect lights; flat gray (or colors) only",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="gaussian_structure.mp4",
        help="Output video path (orbit/cameras_json)",
    )

    args = parser.parse_args()

    if args.model_path:
        model_dir = Path(args.model_path).resolve()
    else:
        model_dir = None

    if args.ply:
        ply_path = Path(args.ply).resolve()
    elif model_dir:
        ply_path = resolve_ply(model_dir, None, args.iteration)
    else:
        parser.error("Provide --ply and/or -m/--model_path")

    print(f"Loading {ply_path}")
    xyz, scales, rotations = load_gaussians(str(ply_path))
    cap_msg = "all" if args.max_gaussians <= 0 else str(args.max_gaussians)
    print(f"  {len(xyz)} Gaussians in file; visualizing up to {cap_msg}")

    per_rgb = None
    if args.use_gaussian_color:
        per_rgb = load_gaussian_dc_rgb(str(ply_path))
        if per_rgb is None:
            print(
                "  [warn] --use_gaussian_color but PLY has no f_dc_0..2; using gray",
                file=sys.stderr,
            )

    mesh = build_combined_ellipsoid_mesh(
        xyz,
        scales,
        rotations,
        max_gaussians=args.max_gaussians,
        sphere_resolution=args.sphere_resolution,
        scale_multiplier=args.scale_multiplier,
        seed=args.seed,
        per_gaussian_rgb=per_rgb,
    )

    if not args.video:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        o3d.visualization.draw_geometries(
            [mesh, frame],
            window_name="3DGS Gaussian Structure",
            mesh_show_back_face=True,
        )
        return 0

    # ---- video ----
    if not model_dir:
        model_dir = ply_path.parent.parent.parent  # .../point_cloud/iter/point_cloud.ply
        if not (model_dir / "cameras.json").is_file():
            parser.error(
                "Video mode needs cameras.json for intrinsics; pass -m <model_dir> "
                "or put cameras.json next to the run."
            )

    cam_json = Path(args.cameras_json) if args.cameras_json else model_dir / "cameras.json"
    if not cam_json.is_file():
        raise FileNotFoundError(cam_json)

    w, h, fx, fy = default_intrinsic_from_cameras_json(
        cam_json, 
        override_width=args.width,
        override_height=args.height
    )

    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)
    bbox_center = (bbox_min + bbox_max) * 0.5
    bbox_diagonal = float(np.linalg.norm(bbox_max - bbox_min))

    if args.trajectory == "orbit":
        poses_c2w = generate_camera_poses(
            args.n_frames, 
            static_intro=args.orbit_static_intro,
            loopable=not args.no_loop
        )
        # Camera positions on the orbit (same as render_video); look-at targets bbox.
        frames_eye = [np.asarray(p[:3, 3], dtype=np.float64) for p in poses_c2w]
        frames_intr = [(w, h, fx, fy)] * len(frames_eye)
    else:
        frames_w2c, frames_intr = load_cameras_json(str(cam_json))
        if len(frames_w2c) == 0:
            raise RuntimeError("No cameras in JSON")
        if args.max_camera_frames is not None:
            frames_w2c = frames_w2c[: args.max_camera_frames]
            frames_intr = frames_intr[: args.max_camera_frames]
        frames_eye = []
        for W2C in frames_w2c:
            R_w2c = np.asarray(W2C[:3, :3], dtype=np.float64)
            t_w2c = np.asarray(W2C[:3, 3], dtype=np.float64)
            frames_eye.append(camera_center_from_opencv_w2c(R_w2c, t_w2c))

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (model_dir / out_path).resolve() if model_dir else Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    render_frames_to_video(
        mesh,
        frames_eye,
        bbox_center,
        frames_intr,
        str(out_path),
        fps=args.fps,
        bbox_diagonal=bbox_diagonal,
        lit=not args.unlit,
        use_vertex_albedo=args.use_gaussian_color and per_rgb is not None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
