#!/usr/bin/env python3
"""
Render train and test views for a trained scene using cameras from the dataset.

Reads ``transforms_train.json`` / ``transforms_test.json`` under a multiview
folder (default ``multiview_static``) on ``source_path`` — not ``cameras.json``
from the model output.

Writes 3DGS-style folders::

    <model_path>/train/ours_<iteration>/renders/*.png
    <model_path>/train/ours_<iteration>/gt/*.png
    <model_path>/test/ours_<iteration>/renders/*.png
    <model_path>/test/ours_<iteration>/gt/*.png

Articulation checkpoints come from ``ckpts/ours_<iteration>.pth``; Gaussians from
``point_cloud/iteration_<iteration>/point_cloud.ply``.

Example::

    python render.py -m output/Table_34610
    python render.py -m output/Table_34610 --iteration 100000
    python render.py -m output/Table_34610 --multiview-dir multiview_static_start
    python render.py -m output/Table_34610 -s /path/to/Table_34610 --skip-train
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene.dataset_readers import readCamerasFromTransformsPNV
from utils.camera_utils import cameraList_from_camInfos
from utils.general_utils import build_rotation, safe_state
from utils.results_json import read_best_iteration


def _resolve_iteration(model_path: Path, override: int | None) -> int:
    """Explicit ``--iteration`` > results.json best > latest ``point_cloud/iteration_*``."""
    if override is not None and override > 0:
        return override

    bi = read_best_iteration(model_path)
    if bi is not None:
        return bi

    pc_root = model_path / "point_cloud"
    iters = sorted(
        int(p.name.split("_")[-1])
        for p in pc_root.glob("iteration_*")
        if p.is_dir()
    )
    if not iters:
        raise RuntimeError(f"No point_cloud/iteration_* under {pc_root}")
    return iters[-1]


def _forced_time_for_dir(multiview_dir: str, explicit: float | None) -> float:
    if explicit is not None:
        return explicit
    name = multiview_dir.rstrip("/").lower()
    if "start" in name:
        return 0.0
    return 1.0


def _load_cameras(
    multiview_path: Path,
    transforms_name: str,
    white_background: bool,
    forced_time: float,
    dataset_args,
) -> list:
    cam_infos = readCamerasFromTransformsPNV(
        str(multiview_path),
        transforms_name,
        white_background,
        extension=".png",
        forced_time=forced_time,
    )
    if not cam_infos:
        return []
    return cameraList_from_camInfos(cam_infos, resolution_scale=1.0, args=dataset_args)


def _render_split(
    views: list,
    *,
    model_path: Path,
    split_name: str,
    iteration: int,
    gaussians: GaussianModel,
    pipeline,
    background: torch.Tensor,
    articulation_weights: torch.Tensor,
    articulation_matrix: torch.Tensor,
    articulation_trans: torch.Tensor,
) -> int:
    if not views:
        print(f"[render] No cameras for split '{split_name}', skipping.")
        return 0

    base = model_path / split_name / f"ours_{iteration}"
    render_dir = base / "renders"
    gt_dir = base / "gt"
    render_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    for view in tqdm(views, desc=f"Rendering {split_name}"):
        pkg = render(
            view,
            gaussians,
            pipeline,
            background,
            articulation_weights=articulation_weights,
            articulation_matrix=articulation_matrix,
            articulation_trans=articulation_trans,
        )
        rendering = torch.clamp(pkg["render"], 0.0, 1.0).cpu()
        gt = torch.clamp(view.original_image[:3], 0.0, 1.0).cpu()

        out_name = f"{view.image_name}.png"
        torchvision.utils.save_image(rendering, render_dir / out_name)
        torchvision.utils.save_image(gt, gt_dir / out_name)

    print(f"[render] {split_name}: {len(views)} views → {render_dir}")
    return len(views)


def render_sets(args) -> int:
    model_path = Path(args.model_path).resolve()
    dataset = ModelParams_for_args.extract(args)
    pipeline = PipelineParams_for_args.extract(args)

    iteration = _resolve_iteration(
        model_path,
        args.iteration if getattr(args, "iteration", -1) > 0 else None,
    )

    ply_path = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    ckpt_path = model_path / "ckpts" / f"ours_{iteration}.pth"
    if not ply_path.is_file():
        print(f"[ERROR] missing {ply_path}", file=sys.stderr)
        return 1
    if not ckpt_path.is_file():
        print(f"[ERROR] missing {ckpt_path}", file=sys.stderr)
        return 1

    source_path = Path(dataset.source_path)
    multiview_dir = getattr(args, "multiview_dir", "multiview_static")
    multiview_path = source_path / multiview_dir
    if not multiview_path.is_dir():
        print(f"[ERROR] multiview folder not found: {multiview_path}", file=sys.stderr)
        return 1

    forced_time = _forced_time_for_dir(
        multiview_dir,
        getattr(args, "forced_time", None),
    )

    print(f"Rendering {model_path}")
    print(f"  iteration={iteration}")
    print(f"  cameras from {multiview_path} (forced_time={forced_time})")

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        gaussians.load_ply(str(ply_path))

        ckpt = torch.load(str(ckpt_path), map_location="cuda")
        art_params = ckpt["articulation_params"]
        art_R = art_params["art_R"].cuda()
        art_T = art_params["art_T"].cuda()
        articulation_matrix = build_rotation(art_R)

        articulation_weights = gaussians.get_weight
        max_indices = torch.argmax(articulation_weights, dim=1)
        hardened = torch.zeros_like(articulation_weights)
        hardened[torch.arange(articulation_weights.shape[0]), max_indices] = 1.0

        bg = [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0]
        background = torch.tensor(bg, dtype=torch.float32, device="cuda")

        train_views = []
        test_views = []
        if not getattr(args, "skip_train", False):
            train_views = _load_cameras(
                multiview_path,
                "transforms_train.json",
                dataset.white_background,
                forced_time,
                dataset,
            )
        if not getattr(args, "skip_test", False):
            test_views = _load_cameras(
                multiview_path,
                "transforms_test.json",
                dataset.white_background,
                forced_time,
                dataset,
            )

        n_train = _render_split(
            train_views,
            model_path=model_path,
            split_name="train",
            iteration=iteration,
            gaussians=gaussians,
            pipeline=pipeline,
            background=background,
            articulation_weights=hardened,
            articulation_matrix=articulation_matrix,
            articulation_trans=art_T,
        )
        n_test = _render_split(
            test_views,
            model_path=model_path,
            split_name="test",
            iteration=iteration,
            gaussians=gaussians,
            pipeline=pipeline,
            background=background,
            articulation_weights=hardened,
            articulation_matrix=articulation_matrix,
            articulation_trans=art_T,
        )

    if n_train == 0 and n_test == 0:
        print("[ERROR] no train or test cameras rendered", file=sys.stderr)
        return 1
    return 0


ModelParams_for_args: ModelParams
PipelineParams_for_args: PipelineParams


def main() -> int:
    global ModelParams_for_args, PipelineParams_for_args
    parser = ArgumentParser(description="Render train/test views from dataset transforms.")
    ModelParams_for_args = ModelParams(parser, sentinel=True)
    PipelineParams_for_args = PipelineParams(parser)
    parser.add_argument(
        "--iteration",
        default=-1,
        type=int,
        help="Checkpoint iteration (default: results.json best, else latest PLY).",
    )
    parser.add_argument(
        "--multiview-dir",
        default="multiview_static",
        type=str,
        help="Multiview folder under source_path (default: multiview_static / end state).",
    )
    parser.add_argument(
        "--forced-time",
        default=None,
        type=float,
        help="Override camera time for articulation (default: 0.0 if dir name contains "
        "'start', else 1.0).",
    )
    parser.add_argument("--skip_train", action="store_true", help="Skip train split.")
    parser.add_argument("--skip_test", action="store_true", help="Skip test split.")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    safe_state(args.quiet)
    return render_sets(args)


if __name__ == "__main__":
    raise SystemExit(main())
