#!/usr/bin/env python3
"""
Render a video of GT meshes interpolating between MPArt start/ end poses with a
single static camera chosen to view articulated motion at ~45° to the mean
motion direction (avoids motion along the view axis and purely edge-on views).

Requires: torch, PyTorch3D (see project README), opencv-python, numpy.

Example:
  python scripts/mesh_interpolation_video.py \\
    --scene_dir data/Table_34610 \\
    --output output/mesh_gt_Table_34610.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def _require_pytorch3d():
    try:
        import pytorch3d  # noqa: F401
    except ImportError as e:
        print(
            "PyTorch3D is required. Install per GaussianArt README (v0.7.5), e.g.\n"
            "  conda activate gaussianart\n"
            "  # then install pytorch3d for your CUDA / torch version\n",
            file=sys.stderr,
        )
        raise e


def load_obj_verts_faces(path: Path, device: torch.device):
    from pytorch3d.io import load_obj

    verts, faces, _aux = load_obj(str(path), device=device, load_textures=False)
    if faces.verts_idx is None or faces.verts_idx.numel() == 0:
        raise ValueError(f"No faces in {path}")
    faces_idx = faces.verts_idx.to(torch.int64)
    return verts, faces_idx


def load_obj_with_texture_atlas(
    path: Path,
    device: torch.device,
    atlas_size: int,
):
    """
    Load OBJ + MTL; build a per-face texture atlas so all map_Kd materials apply.
    PyTorch3D's load_objs_as_meshes only uses the first map — multi-material meshes
    need create_texture_atlas=True.
    """
    from pytorch3d.io import load_obj

    verts, faces, aux = load_obj(
        str(path),
        device=device,
        load_textures=True,
        create_texture_atlas=True,
        texture_atlas_size=atlas_size,
        texture_wrap="repeat",
    )
    if faces.verts_idx is None or faces.verts_idx.numel() == 0:
        raise ValueError(f"No faces in {path}")
    if aux.texture_atlas is None:
        raise ValueError(f"texture_atlas missing for {path} (create_texture_atlas failed)")
    return verts, faces, aux.texture_atlas


def discover_mesh_stems(gt_dir: Path) -> list[str]:
    start_dir = gt_dir / "start"
    stems = []
    for p in sorted(start_dir.glob("*.obj")):
        name = p.stem
        if name.endswith("_total"):
            continue
        if not (gt_dir / "end" / f"{name}.obj").exists():
            continue
        stems.append(name)
    if not stems:
        raise FileNotFoundError(f"No matching start/end OBJ pairs under {gt_dir}")
    return stems


def merge_meshes(
    verts_list: list[torch.Tensor],
    faces_list: list[torch.Tensor],
    colors_list: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate meshes; colors per vertex [V,3]."""
    v_offset = 0
    all_v = []
    all_f = []
    all_c = []
    for v, f, c in zip(verts_list, faces_list, colors_list):
        all_v.append(v)
        all_c.append(c)
        all_f.append(f + v_offset)
        v_offset += v.shape[0]
    verts = torch.cat(all_v, dim=0)
    faces = torch.cat(all_f, dim=0)
    colors = torch.cat(all_c, dim=0)
    return verts, faces, colors


def mean_motion_direction(
    start_dir: Path,
    end_dir: Path,
    stems: list[str],
    device: torch.device,
    static_name: str = "static",
) -> torch.Tensor:
    """Dominant translation direction from dynamic parts (vertex displacement)."""
    deltas = []
    weights = []
    for stem in stems:
        if stem == static_name:
            continue
        v0, _ = load_obj_verts_faces(start_dir / f"{stem}.obj", device)
        v1, _ = load_obj_verts_faces(end_dir / f"{stem}.obj", device)
        if v0.shape != v1.shape:
            raise ValueError(f"Topology mismatch for {stem}: {v0.shape} vs {v1.shape}")
        d = (v1 - v0).mean(dim=0)
        w = float(v0.shape[0])
        deltas.append(d * w)
        weights.append(w)
    if not deltas:
        for stem in stems:
            if stem != static_name:
                v0, _ = load_obj_verts_faces(start_dir / f"{stem}.obj", device)
                v1, _ = load_obj_verts_faces(end_dir / f"{stem}.obj", device)
                d = (v1 - v0).mean(dim=0)
                return d / (d.norm() + 1e-8)
        return torch.tensor([0.0, -1.0, 0.0], device=device, dtype=torch.float32)

    m = torch.stack(deltas, dim=0).sum(dim=0) / (sum(weights) + 1e-8)
    n = m.norm()
    if n < 1e-6:
        return torch.tensor([0.0, -1.0, 0.0], device=device, dtype=torch.float32)
    return m / n


def fallback_axis_from_trans(gt_dir: Path, device: torch.device) -> torch.Tensor | None:
    p = gt_dir / "trans.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    infos = data.get("trans_info") or []
    if not infos:
        return None
    d = np.array(infos[0]["axis"]["d"], dtype=np.float32)
    v = torch.tensor(d, device=device, dtype=torch.float32)
    return v / (v.norm() + 1e-8)


def camera_ray_oblique_to_motion(
    motion: torch.Tensor,
    scene_center: torch.Tensor,
    distance: float,
    angle_deg: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Place camera so the view direction (scene_center -> camera, outward) is
    oblique to `motion`: the inward view direction (camera -> scene) makes
    `angle_deg` with normalized motion vector `m`. This avoids cameras whose
    view axis is parallel to motion (invisible 2D flow) and reduces fully
    edge-on setups by mixing motion with a perpendicular vector.
    """
    m = motion / (motion.norm() + 1e-8)
    up_ref = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)
    if torch.abs(torch.dot(m, up_ref)) > 0.95:
        up_ref = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32)
    side = torch.linalg.cross(m.unsqueeze(0), up_ref.unsqueeze(0)).squeeze(0)
    side = side / (side.norm() + 1e-8)

    theta = math.radians(angle_deg)
    # Inward = direction from camera toward scene center (what look-at uses).
    # Angle between inward and m equals theta when inward = cos(t)*m + sin(t)*side (side ⟂ m).
    inward = math.cos(theta) * m + math.sin(theta) * side
    inward = inward / (inward.norm() + 1e-8)

    eye = scene_center - distance * inward
    return eye, scene_center


def fov_deg_from_intrinsics(fx: float, image_width: float) -> float:
    return math.degrees(2.0 * math.atan(image_width / (2.0 * fx)))


def build_renderer(
    faces: torch.Tensor,
    cameras,
    lights,
    materials,
    image_size: tuple[int, int],
    device: torch.device,
    vert_colors: torch.Tensor | None = None,
    textures_atlas=None,
):
    from pytorch3d.renderer import (
        Materials,
        MeshRasterizer,
        MeshRenderer,
        RasterizationSettings,
        SoftPhongShader,
        TexturesVertex,
    )
    from pytorch3d.structures import Meshes

    if vert_colors is None and textures_atlas is None:
        raise ValueError("Need vert_colors or textures_atlas")

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    # Phong combines (ambient+diffuse)*texels + specular; Materials defaults are fully
    # specular-capable. Pass explicit materials so "flat" mode is true albedo-only.
    shader = SoftPhongShader(
        device=device, cameras=cameras, lights=lights, materials=materials
    )
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

    def render_one(verts: torch.Tensor):
        if textures_atlas is not None:
            mesh = Meshes(verts=[verts], faces=[faces], textures=textures_atlas)
        else:
            tex = TexturesVertex(verts_features=vert_colors.unsqueeze(0))
            mesh = Meshes(verts=[verts], faces=[faces], textures=tex)
        img = renderer(mesh)
        return img[0, ..., :3].clamp(0.0, 1.0)

    return render_one


def main():
    _require_pytorch3d()

    from pytorch3d.renderer import (
        FoVPerspectiveCameras,
        PointLights,
        TexturesAtlas,
        look_at_view_transform,
    )
    from pytorch3d.structures import Meshes, join_meshes_as_batch, join_meshes_as_scene

    parser = argparse.ArgumentParser(description="MPArt GT mesh interpolation video (PyTorch3D)")
    parser.add_argument(
        "--scene_dir",
        type=Path,
        default=Path("data/Table_34610"),
        help="Path to scene folder (contains gt/start, gt/end, transforms_train_start.json)",
    )
    parser.add_argument("--output", type=Path, default=Path("output/mesh_gt_video.mp4"))
    parser.add_argument("--num_frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument(
        "--motion_angle_deg",
        type=float,
        default=45.0,
        help="Angle between view ray (into scene) and mean motion direction",
    )
    parser.add_argument("--image_size", type=int, nargs=2, default=[800, 800])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--texture_atlas_size",
        type=int,
        default=64,
        help="Per-face texture resolution in the material atlas (higher = sharper, more VRAM)",
    )
    parser.add_argument(
        "--vertex_colors_only",
        action="store_true",
        help="Skip MTL textures; use solid per-part vertex colors (faster, for debugging)",
    )
    parser.add_argument(
        "--camera_lift",
        type=float,
        default=0.35,
        help="Move camera upward (+Y in scene units) after placement for a higher viewpoint",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    gt_dir = args.scene_dir / "gt"
    start_dir = gt_dir / "start"
    end_dir = gt_dir / "end"
    stems = discover_mesh_stems(gt_dir)

    # Intrinsics from transforms (same as training data)
    transforms_path = args.scene_dir / "transforms_train_start.json"
    if not transforms_path.exists():
        transforms_path = args.scene_dir / "transforms_train.json"
    fx = 1111.1113654242622
    if transforms_path.exists():
        with open(transforms_path) as f:
            tj = json.load(f)
        fr0 = tj["frames"][0]
        if "camera_intrinsics" in fr0:
            K = fr0["camera_intrinsics"]
            fx = float(K[0][0])
    W, H = int(args.image_size[0]), int(args.image_size[1])
    fov = fov_deg_from_intrinsics(fx, float(W))

    # Per-part vertex colors (fallback / debug)
    rng_colors = torch.tensor(
        [
            [0.75, 0.35, 0.25],
            [0.25, 0.55, 0.85],
            [0.35, 0.75, 0.35],
            [0.85, 0.75, 0.25],
            [0.65, 0.35, 0.75],
            [0.45, 0.45, 0.50],
        ],
        device=device,
        dtype=torch.float32,
    )

    f_cat: torch.Tensor
    v_start_cat: torch.Tensor
    v_end_cat: torch.Tensor
    vert_colors: torch.Tensor | None = None
    textures_atlas = None

    if not args.vertex_colors_only:
        try:
            parts: list = []
            verts_end_list: list[torch.Tensor] = []
            for stem in stems:
                v0, faces_s, atlas = load_obj_with_texture_atlas(
                    start_dir / f"{stem}.obj",
                    device,
                    args.texture_atlas_size,
                )
                v1, faces_e = load_obj_verts_faces(end_dir / f"{stem}.obj", device)
                if v0.shape != v1.shape or faces_s.verts_idx.shape != faces_e.shape:
                    raise ValueError(f"Topology mismatch for {stem}")
                tex = TexturesAtlas(atlas=[atlas.to(device)])
                parts.append(
                    Meshes(
                        verts=[v0],
                        faces=[faces_s.verts_idx.to(torch.int64)],
                        textures=tex,
                    )
                )
                verts_end_list.append(v1)

            scene_mesh = join_meshes_as_scene(join_meshes_as_batch(parts), include_textures=True)
            v_start_cat = scene_mesh.verts_list()[0]
            f_cat = scene_mesh.faces_list()[0]
            textures_atlas = scene_mesh.textures
            v_end_cat = torch.cat(verts_end_list, dim=0)
            if v_start_cat.shape[0] != v_end_cat.shape[0]:
                raise RuntimeError("Packed vertex count mismatch between textured scene and end verts")
            print(
                f"Loaded textured mesh: {v_start_cat.shape[0]} verts, {f_cat.shape[0]} faces, "
                f"atlas cell {args.texture_atlas_size}x{args.texture_atlas_size}"
            )
        except Exception as e:
            print(f"[warn] Textured load failed ({e}); falling back to vertex colors.", file=sys.stderr)
            args.vertex_colors_only = True

    if args.vertex_colors_only:
        verts_start = []
        verts_end = []
        faces_parts = []
        colors_parts = []
        for i, stem in enumerate(stems):
            v0, f0 = load_obj_verts_faces(start_dir / f"{stem}.obj", device)
            v1, f1 = load_obj_verts_faces(end_dir / f"{stem}.obj", device)
            if v0.shape != v1.shape or f0.shape != f1.shape:
                raise ValueError(f"Mismatch for {stem}: verts or faces differ between start/end")
            c = rng_colors[i % rng_colors.shape[0]].expand(v0.shape[0], -1)
            verts_start.append(v0)
            verts_end.append(v1)
            faces_parts.append(f0)
            colors_parts.append(c)

        _v0, f_cat, vert_colors = merge_meshes(verts_start, faces_parts, colors_parts)
        _v1, _, _ = merge_meshes(verts_end, faces_parts, colors_parts)
        del _v0, _v1

        v_start_cat = torch.cat(verts_start, dim=0)
        v_end_cat = torch.cat(verts_end, dim=0)
        textures_atlas = None

    motion = mean_motion_direction(start_dir, end_dir, stems, device)
    fb = fallback_axis_from_trans(gt_dir, device)
    if motion.norm() < 1e-4 and fb is not None:
        motion = fb
    elif motion.norm() < 1e-4:
        motion = torch.tensor([0.0, -1.0, 0.0], device=device)

    # Scene center and scale from midpoint geometry (same vertex order as merged mesh)
    v_mid = 0.5 * (v_start_cat + v_end_cat)
    center = v_mid.mean(dim=0)
    radius = (v_mid - center).norm(dim=1).max().item()
    dist = max(radius * 2.8, 1.5)

    eye, at = camera_ray_oblique_to_motion(
        motion, center, dist, args.motion_angle_deg, device
    )
    lift = torch.tensor([0.0, float(args.camera_lift), 0.0], device=device, dtype=torch.float32)
    eye = eye + lift

    # First positional arg to look_at_view_transform is `dist`, not `eye` — pass eye= explicitly.
    up_world = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32, device=device)
    R, T = look_at_view_transform(
        eye=eye.unsqueeze(0),
        at=at.unsqueeze(0),
        up=up_world,
        device=device,
    )
    try:
        cameras = FoVPerspectiveCameras(
            device=device,
            R=R,
            T=T,
            fov=fov,
            znear=0.01,
            zfar=500.0,
            image_size=((H, W),),
        )
    except TypeError:
        cameras = FoVPerspectiveCameras(
            device=device,
            R=R,
            T=T,
            fov=fov,
            znear=0.01,
            zfar=500.0,
        )
    to_cam = eye - center
    to_cam = to_cam / (to_cam.norm() + 1e-8)
    up = torch.tensor([0.0, 1.0, 0.0], device=device)
    fill = torch.linalg.cross(to_cam.unsqueeze(0), up.unsqueeze(0)).squeeze(0)
    fill = fill / (fill.norm() + 1e-8)
    light_pos = eye + 0.35 * (center - eye) + 0.45 * fill
    lights = PointLights(
        device=device,
        location=light_pos.unsqueeze(0),
        ambient_color=((0.5, 0.5, 0.5),),
        diffuse_color=((0.75, 0.75, 0.75),),
        specular_color=((0.2, 0.2, 0.2),),
    )

    render_one = build_renderer(
        f_cat,
        cameras,
        lights,
        (H, W),
        device,
        vert_colors=vert_colors,
        textures_atlas=textures_atlas,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames_np: list[np.ndarray] = []

    for fi in range(args.num_frames):
        t = fi / max(args.num_frames - 1, 1)
        verts = (1.0 - t) * v_start_cat + t * v_end_cat
        img = render_one(verts)
        rgb = (img.cpu().numpy() * 255.0).astype(np.uint8)
        frames_np.append(rgb)

    # Write video: prefer imageio-ffmpeg; fallback OpenCV; fallback PNG + ffmpeg CLI
    try:
        import imageio.v2 as imageio

        try:
            imageio.mimsave(
                str(args.output),
                frames_np,
                fps=args.fps,
                codec="libx264",
                quality=8,
            )
        except (TypeError, ValueError):
            imageio.mimsave(str(args.output), frames_np, fps=args.fps)
        print(f"Wrote {args.output} ({len(frames_np)} frames @ {args.fps} fps)")
    except Exception:
        try:
            import cv2

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(args.output), fourcc, args.fps, (W, H))
            for fr in frames_np:
                out.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            out.release()
            print(f"Wrote {args.output} with OpenCV ({len(frames_np)} frames)")
        except Exception as e1:
            stem = args.output.with_suffix("")
            tmp = stem.parent / f"{stem.name}_frames"
            tmp.mkdir(parents=True, exist_ok=True)
            import cv2

            for i, fr in enumerate(frames_np):
                cv2.imwrite(str(tmp / f"{i:05d}.png"), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(args.fps),
                "-i",
                str(tmp / "%05d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(args.output),
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                print(f"Wrote {args.output} via ffmpeg from {tmp}")
            except Exception as e2:
                print(f"imageio failed: {e1}; opencv mp4 failed; ffmpeg failed: {e2}")
                print(f"Saved PNG sequence under {tmp}")
                sys.exit(1)


if __name__ == "__main__":
    main()
