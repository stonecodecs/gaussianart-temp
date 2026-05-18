#!/usr/bin/env python3
"""
Render the same orbital articulation video as ``render_video.py``, with predicted
(and optional GT) joint axes drawn on top.

Uses the 3D Gaussian rasterizer (not Open3D) for geometry, then projects axis
segments into each frame with the same ``full_proj_transform`` as training.

Output::

    <model_path>/video/ours_<iteration>/axis-output.mp4

Usage::

    python scripts/render_axis_video.py -m output/PNV/Bucket_100443
    python scripts/render_axis_video.py -m output/PNV/Bucket_100443 --N-frames 60
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from arguments import ModelParams, PipelineParams  # noqa: E402
from eval_axis import interpret_transforms, read_gt  # noqa: E402
from gaussian_renderer import GaussianModel, render  # noqa: E402
from render_video import generate  # noqa: E402
from scene import Scene  # noqa: E402
from utils.general_utils import build_rotation, safe_state  # noqa: E402
from utils.results_json import load_results_json, read_best_iteration  # noqa: E402
from utils.rotation_utils import R_from_axis_angle, rotation_matrices_to_axes_angles  # noqa: E402
from utils.system_utils import searchForMaxIteration  # noqa: E402

# BGR for cv2
PRED_COLORS_BGR = [
    (23, 126, 232),
    (24, 228, 58),
    (238, 18, 55),
    (0, 255, 255),
    (42, 42, 165),
    (238, 130, 238),
    (242, 191, 140),
]
GT_COLOR_BGR = (30, 30, 30)


def load_cfg_source_path(model_dir: Path) -> Path | None:
    cfg = model_dir / "cfg_args"
    if not cfg.is_file():
        return None
    text = cfg.read_text()
    m = re.search(r"source_path='([^']+)'", text) or re.search(r'source_path="([^"]+)"', text)
    return Path(m.group(1)) if m else None


def resolve_gt_trans_path(model_dir: Path) -> Path | None:
    src = load_cfg_source_path(model_dir)
    if src is None:
        return None
    for rel in (Path("singleview_dynamic/gt/trans.json"), Path("gt/trans.json")):
        cand = src / rel
        if cand.is_file():
            return cand
    return None


def joints_from_checkpoint(ckpt_path: Path) -> list[dict]:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    art_R = ckpt["articulation_params"]["art_R"]
    art_T = ckpt["articulation_params"]["art_T"]
    articulation_matrix = build_rotation(art_R).detach().cpu().numpy()
    art_T_np = art_T.detach().cpu().numpy()
    n = articulation_matrix.shape[0]
    joints: list[dict] = []
    # Part 0 is the frozen root / static frame when n > 2 (see freeze_parts in run.py /
    # train_eval_all_pnv.py). Part n-1 is the extra static slot and is not in range(n-1).
    # For 2-part scenes the mover is index 0 and static is index 1.
    start_idx = 1 if n > 2 else 0
    for i in range(start_idx, n - 1):
        cur_R = articulation_matrix[i]
        cur_T = art_T_np[i]
        joint_type = "prismatic" if float(np.abs(cur_R - np.eye(3)).mean()) < 5e-2 else "revolute"
        pred_joint, _, _ = interpret_transforms(
            np.eye(3), np.zeros(3), cur_R, cur_T, joint_type=joint_type
        )
        pred_joint["joint_type"] = joint_type
        joints.append(pred_joint)
    return joints


def joints_from_gt(gt_path: Path, legacy: bool = False) -> list[dict]:
    raw = read_gt(str(gt_path), legacy=legacy)
    joints: list[dict] = []
    for j in raw:
        jtype = j.get("type", "revolute")
        if jtype == "rotate":
            jtype = "revolute"
        joints.append(
            {
                "axis_position": np.asarray(j["axis_position"], dtype=np.float64),
                "axis_direction": np.asarray(j["axis_direction"], dtype=np.float64),
                "joint_type": jtype,
            }
        )
    return joints


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return v / n


def project_world_to_pixels(
    pts_world: np.ndarray,
    view,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project world points to pixel coordinates (same convention as ``loss_utils.world2pix``).
    Returns (x, y, valid) with x,y shape (N,).
    """
    w, h = int(view.image_width), int(view.image_height)
    device = view.full_proj_transform.device
    pts = torch.tensor(pts_world, dtype=torch.float32, device=device)
    if pts.ndim == 1:
        pts = pts.unsqueeze(0)
    ones = torch.ones((pts.shape[0], 1), dtype=torch.float32, device=device)
    pts4 = torch.cat([pts, ones], dim=1)
    p_hom = pts4 @ view.full_proj_transform
    w_hom = p_hom[:, 3]
    valid = w_hom > 1e-6
    p_proj = p_hom[:, :3] / w_hom.unsqueeze(1).clamp(min=1e-8)
    x = ((p_proj[:, 0] + 1.0) * (w - 1) + 1) * 0.5
    y = ((p_proj[:, 1] + 1.0) * (h - 1) + 1) * 0.5
    return x.cpu().numpy(), y.cpu().numpy(), valid.cpu().numpy()


def draw_axes_on_bgr(
    img_bgr: np.ndarray,
    joints: list[dict],
    view,
    colors_bgr: list[tuple[int, int, int]] | None,
    default_color_bgr: tuple[int, int, int],
    axis_half_len: float,
    line_thickness: int = 3,
) -> np.ndarray:
    """Draw axis lines and origin markers on a BGR uint8 image (in place)."""
    h, w = img_bgr.shape[:2]
    for j, joint in enumerate(joints):
        color = (
            colors_bgr[j % len(colors_bgr)]
            if colors_bgr
            else default_color_bgr
        )
        o = np.asarray(joint["axis_position"], dtype=np.float64)
        d = _normalize(np.asarray(joint["axis_direction"], dtype=np.float64))
        p0 = o - d * axis_half_len
        p1 = o + d * axis_half_len
        pts3d = np.stack([p0, o, p1], axis=0)
        xs, ys, valid = project_world_to_pixels(pts3d, view)
        xi = np.round(xs).astype(np.int32)
        yi = np.round(ys).astype(np.int32)

        if valid[0] and valid[2]:
            cv2.line(
                img_bgr,
                (int(np.clip(xi[0], 0, w - 1)), int(np.clip(yi[0], 0, h - 1))),
                (int(np.clip(xi[2], 0, w - 1)), int(np.clip(yi[2], 0, h - 1))),
                color,
                line_thickness,
                cv2.LINE_AA,
            )
        if valid[1]:
            cx, cy = int(np.clip(xi[1], 0, w - 1)), int(np.clip(yi[1], 0, h - 1))
            cv2.circle(img_bgr, (cx, cy), max(line_thickness + 2, 5), color, -1, cv2.LINE_AA)
            cv2.circle(img_bgr, (cx, cy), max(line_thickness + 2, 5), (255, 255, 255), 1, cv2.LINE_AA)
    return img_bgr


def write_video_bgr(frames_bgr: list[np.ndarray], path: Path, fps: float) -> None:
    if not frames_bgr:
        return
    h, w = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames_bgr:
        writer.write(frame)
    writer.release()
    print(f"Wrote video: {path}")


def load_cfg_namespace(model_path: Path, iteration: int) -> object:
    """Load merged args from ``cfg_args`` (same fields as ``get_combined_args``)."""
    from argparse import Namespace

    cfg_path = model_path / "cfg_args"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing {cfg_path}")
    with cfg_path.open() as f:
        cfg_ns = eval(f.read())
    merged = vars(cfg_ns).copy()
    merged["model_path"] = str(model_path.resolve())
    merged["iteration"] = iteration
    return Namespace(**merged)


def resolve_iteration(model_dir: Path, iteration: int | None) -> int:
    if iteration is not None and iteration >= 0:
        return iteration
    bi = read_best_iteration(model_dir)
    if bi is not None:
        return bi
    pc_root = model_dir / "point_cloud"
    if pc_root.is_dir():
        return searchForMaxIteration(str(pc_root))
    ckpts = sorted(model_dir.glob("ckpts/ours_*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No ckpts under {model_dir / 'ckpts'}")
    return int(ckpts[-1].stem.split("_")[-1])


def render_axis_video(
    model_path: Path,
    iteration: int | None = None,
    n_frames: int = 60,
    fps: float = 30.0,
    legacy_gt: bool = False,
    draw_gt: bool = True,
) -> Path:
    """
    Same orbit + articulation schedule as ``render_video.render_set``, with axis overlay.
    """
    model_path = model_path.resolve()
    iter_ = resolve_iteration(model_path, iteration)
    ckpt_path = model_path / "ckpts" / f"ours_{iter_}.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    pred_joints = joints_from_checkpoint(ckpt_path)
    if not pred_joints:
        raise RuntimeError(f"No joints in {ckpt_path}")

    gt_joints: list[dict] = []
    if draw_gt:
        gt_path = resolve_gt_trans_path(model_path)
        if gt_path is not None:
            results = load_results_json(model_path)
            legacy = legacy_gt or bool(
                isinstance(results.get("axis_eval"), dict)
                and results["axis_eval"].get("legacy_coord")
            )
            gt_joints = joints_from_gt(gt_path, legacy=legacy)
            print(f"GT axes: {gt_path}")
        else:
            print("GT trans.json not found (predicted axes only).")

    # --- 3DGS scene (mirrors render_video.render_sets) ---
    from argparse import ArgumentParser

    parser = ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    pipe_defaults = PipelineParams(ArgumentParser())
    args = load_cfg_namespace(model_path, iter_)

    with torch.no_grad():
        gaussians = GaussianModel(model_params.extract(args).sh_degree)
        dataset = model_params.extract(args)
        pipeline = pipeline_params.extract(args)
        for key in vars(pipe_defaults):
            if not hasattr(pipeline, key):
                setattr(pipeline, key, getattr(pipe_defaults, key))
        scene = Scene(dataset, gaussians, load_iteration=iter_, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        views = generate(scene.getTrainCameras(), N=n_frames)

        ckpt = torch.load(ckpt_path, map_location="cuda")
        art_R = ckpt["articulation_params"]["art_R"].cuda()
        art_T = ckpt["articulation_params"]["art_T"].cuda()

        articulation_weights = gaussians.get_weight
        max_indices = torch.argmax(articulation_weights, dim=1)
        hardened = torch.zeros_like(articulation_weights)
        hardened[torch.arange(articulation_weights.shape[0]), max_indices] = 1.0
        articulation_weights = hardened

        articulation_matrix = build_rotation(art_R)
        articulation_trans = art_T
        rot_axis, rot_angle = rotation_matrices_to_axes_angles(articulation_matrix)

        # Axis line length from scene extent
        xyz = gaussians.get_xyz.detach().cpu().numpy()
        bbox_diag = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))
        axis_half_len = max(bbox_diag * 0.35, 0.05)

        n_rot_views = max(len(views) // 3, 1)
        ts = torch.cat(
            [
                torch.linspace(0, 1, max(n_rot_views // 2, 1)),
                torch.linspace(1, 0, max(n_rot_views - n_rot_views // 2, 1)),
            ]
            * 3
        )
        if len(ts) < len(views):
            ts = torch.cat([ts, ts[-1:].repeat(len(views) - len(ts))])
        ts = ts[: len(views)]

        frames_bgr: list[np.ndarray] = []
        for idx, view in enumerate(tqdm(views, desc="Axis video (3DGS + overlay)")):
            t = float(ts[idx])
            angle_t = rot_angle * t
            trans_t = articulation_trans * t
            matrix_t = torch.stack(
                [
                    R_from_axis_angle(rot_axis[i], angle_t[i])
                    for i in range(rot_angle.shape[0])
                ]
            ).cuda()

            render_pkg = render(
                view,
                gaussians,
                pipeline,
                background,
                articulation_weights=articulation_weights,
                articulation_matrix=matrix_t,
                articulation_trans=trans_t,
                force_transform=True,
            )
            rgb = render_pkg["render"].detach().clamp(0.0, 1.0)
            img_bgr = (
                (rgb.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            )
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)

            draw_axes_on_bgr(
                img_bgr,
                pred_joints,
                view,
                PRED_COLORS_BGR,
                PRED_COLORS_BGR[0],
                axis_half_len,
                line_thickness=3,
            )
            if gt_joints:
                draw_axes_on_bgr(
                    img_bgr,
                    gt_joints,
                    view,
                    None,
                    GT_COLOR_BGR,
                    axis_half_len,
                    line_thickness=2,
                )
            frames_bgr.append(img_bgr)

    out_dir = model_path / "video" / f"ours_{iter_}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "axis-output.mp4"
    write_video_bgr(frames_bgr, out_path, fps=fps)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Orbital 3DGS video with predicted/GT axis overlay (same motion as render_video.py)"
    )
    parser.add_argument("-m", "--model_path", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--N-frames", type=int, default=72, dest="n_frames")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--legacy-gt", action="store_true")
    parser.add_argument("--no-gt", action="store_true", help="Do not draw GT axes")
    args = parser.parse_args()

    safe_state(True)
    iteration = None if args.iteration < 0 else args.iteration
    try:
        out = render_axis_video(
            args.model_path,
            iteration=iteration,
            n_frames=args.n_frames,
            fps=args.fps,
            legacy_gt=args.legacy_gt,
            draw_gt=not args.no_gt,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
