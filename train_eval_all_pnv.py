#!/usr/bin/env python3
"""
Train, render, and evaluate every scene in a PartNet-Video (data120-style) dataset.

Dataset layout expected per scene::

    <data_root>/<scene_name>/
        multiview_static_start/   ← start-state cameras
        multiview_static/         ← end-state cameras
        singleview_dynamic/gt/trans.json   ← joint ground-truth (motion axes)
        points3d.ply
        semantics.npy

Outputs per scene::

    <output_dir>/<scene_name>/
        ckpts/ours_*.pth           ← articulation checkpoints (from train.py)
        point_cloud/               ← Gaussian splats
        meshes/tsdf_mesh_*.ply     ← TSDF reconstructions (full, static, dynamic)
        results.json               ← merged metrics (axis_eval, image_metrics, point_cloud_stats, chamfer_metrics)
        train/ours_<iter>/renders/ ← dataset train views (render.py)
        test/ours_<iter>/renders/  ← dataset test views (render.py)
        video/                     ← orbital demo MP4 + frames (render_video.py)
        gaussian_structure_orbit.mp4  ← ellipsoid orbit (Open3D, DC colors, all Gaussians)
        error.txt                  ← only written when a stage fails

A CSV summary is appended to ``<output_dir>/train_eval_summary.csv``.

Example::

    python train_eval_all_pnv.py --data-dir data120 --gpu 0
    python train_eval_all_pnv.py --data-dir data120 --models Safe_101605 --skip-train
    python train_eval_all_pnv.py --data-dir data120 --reeval-completed
        # scenes with ours_<iterations>.pth: skip train, re-run eval/render/metrics

Multi-GPU / Kubernetes: run one process per worker with the same ``--num-gpus`` and a
distinct ``--gpu-id``. Scenes (sorted) are split by striding: worker *k* gets
``scenes[k::num_gpus]``. Each worker uses ``CUDA_VISIBLE_DEVICES`` from ``--gpu`` (usually
``0`` when the scheduler assigns one GPU per pod).

Example (4 workers)::

    python train_eval_all_pnv.py --data-dir /data/pnv --num-gpus 4 --gpu-id 0 --gpu 0
    python train_eval_all_pnv.py --data-dir /data/pnv --num-gpus 4 --gpu-id 1 --gpu 0
    # ... gpu-id 2, 3

When ``num-gpus > 1``, the default summary CSV is per-shard
(``train_eval_summary.shard{k}_of_{n}.csv``) to avoid concurrent append corruption.
Pass an explicit ``--summary`` path only if you handle merging yourself.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from utils.results_json import results_json_path, save_results_json_merged
# Helpers
# ---------------------------------------------------------------------------

def _gt_trans_path(scene_dir: Path) -> Path | None:
    """Return the trans.json path for a PartNet-Video scene, or None."""
    candidate = scene_dir / "singleview_dynamic" / "gt" / "trans.json"
    return candidate if candidate.is_file() else None


def list_scenes(data_root: Path) -> list[str]:
    """Return sorted scene names that have the required PartNet-Video layout."""
    if not data_root.is_dir():
        return []
    scenes: list[str] = []
    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "multiview_static").is_dir() and _gt_trans_path(child) is not None:
            scenes.append(child.name)
    return scenes


def _num_parts_from_trans(gt_trans: Path) -> tuple[int, list[int]]:
    """
    Parse trans.json and return (num_parts, freeze_parts) exactly as run.py does.

    num_parts   = len(trans_info) + 1
    freeze_parts = indices of "translate"-type joints + the extra static index
    """
    with gt_trans.open() as f:
        data = json.load(f)
    trans_info = data.get("trans_info", [])
    translate_indices = [
        i for i, item in enumerate(trans_info) if item.get("type") == "translate"
    ]
    translate_indices.append(len(trans_info))   # extra static part
    return len(trans_info) + 1, translate_indices


def _latest_ply(model_path: Path) -> Path | None:
    """Return the most recent point_cloud PLY under model_path, or None."""
    pc_dir = model_path / "point_cloud"
    if not pc_dir.is_dir():
        return None
    candidates = sorted(pc_dir.glob("iteration_*/point_cloud.ply"))
    return candidates[-1] if candidates else None


def _count_gaussians(ply_path: Path) -> int | None:
    """Return the number of Gaussian primitives in a PLY file, or None on error."""
    try:
        from plyfile import PlyData
        ply = PlyData.read(str(ply_path))
        return ply["vertex"].count
    except Exception:
        return None


def _write_error(model_path: Path, scene: str, stage: str, detail: str) -> None:
    """Append an error entry to <model_path>/error.txt."""
    model_path.mkdir(parents=True, exist_ok=True)
    with (model_path / "error.txt").open("a") as f:
        ts = datetime.now(timezone.utc).isoformat()
        f.write(f"[{ts}] {stage} failed for {scene}\n{detail}\n\n")


def _sample_surface(mesh: trimesh.Trimesh, n: int) -> np.ndarray:
    """Sample n points uniformly from the mesh surface."""
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return pts


def _chamfer_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """
    Compute symmetric Chamfer distance between two point clouds.
    Returns mean squared distance (smaller is better).
    """
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)
    d_ab, _ = tree_b.query(pts_a)  # nearest-neighbour distances A→B
    d_ba, _ = tree_a.query(pts_b)  # nearest-neighbour distances B→A
    cd = float(np.mean(d_ab ** 2) + np.mean(d_ba ** 2))
    return cd


def _compute_mesh_chamfer(
    tsdf_mesh_path: Path,
    gt_mesh_path: Path,
    n_samples: int = 10_000
) -> dict | None:
    """
    Compute Chamfer distance between TSDF reconstruction and GT mesh.
    Returns dict with 'chamfer_distance' or None on error.
    """
    try:
        if not tsdf_mesh_path.is_file():
            return None
        if not gt_mesh_path.is_file():
            return None
        
        tsdf_mesh = trimesh.load(str(tsdf_mesh_path), force="mesh", process=False)
        gt_mesh = trimesh.load(str(gt_mesh_path), force="mesh", process=False)
        
        # Ensure meshes have geometry
        if not hasattr(tsdf_mesh, 'vertices') or len(tsdf_mesh.vertices) == 0:
            return None
        if not hasattr(gt_mesh, 'vertices') or len(gt_mesh.vertices) == 0:
            return None
        
        pts_tsdf = _sample_surface(tsdf_mesh, n_samples)
        pts_gt = _sample_surface(gt_mesh, n_samples)
        
        cd = _chamfer_distance(pts_tsdf, pts_gt)
        return {"chamfer_distance": cd}
    except Exception as e:
        print(f"[CHAMFER] Error computing Chamfer distance: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    repo_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Train + render + eval all PartNet-Video scenes in a data directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "data120",
        help="Dataset root containing scene folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/PartNetVideo"),
        help="Training output directory (absolute, or relative to repo root)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="CUDA device index passed to CUDA_VISIBLE_DEVICES for child processes",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        dest="num_gpus",
        help="Total parallel workers; scenes are split across [0, num-gpus)",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        dest="gpu_id",
        help="This worker's index in [0, num-gpus); processes scenes[gpu-id::num-gpus]",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Only process these scene names (default: all valid scenes)",
    )
    parser.add_argument("--iterations", type=int, default=100_000, help="Number of iterations to train")
    parser.add_argument(
        "--max-parts",
        type=int,
        default=21,
        dest="max_parts",
        help="Skip scenes whose trans.json yields more than this many parts (movable + 1 static)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process scenes even when the final checkpoint (ours_<iterations>.pth) already exists",
    )
    parser.add_argument(
        "--reeval-completed",
        action="store_true",
        dest="reeval_completed",
        help="If ours_<iterations>.pth exists, skip training but still run eval, "
             "image metrics, dataset render, orbital video, and structure video "
             "(default: skip the whole scene)",
    )
    parser.add_argument("--skip-train",  action="store_true", help="Skip training")
    parser.add_argument("--skip-eval",   action="store_true", help="Skip axis evaluation")
    parser.add_argument(
        "--skip-image-metrics",
        action="store_true",
        dest="skip_image_metrics",
        help="Skip eval_image_metrics.py (PSNR/SSIM/LPIPS on the test set)",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Skip render_video.py orbital demo (video/)",
    )
    parser.add_argument(
        "--skip-dataset-render",
        action="store_true",
        dest="skip_dataset_render",
        help="Skip render.py train/test views from dataset transforms",
    )
    parser.add_argument(
        "--multiview-dir",
        default="multiview_static",
        help="Multiview folder under each scene passed to render.py",
    )
    parser.add_argument(
        "--skip-structure-video",
        action="store_true",
        help="Skip scripts/vis_gaussian_structure.py orbital ellipsoid MP4",
    )
    parser.add_argument(
        "--skip-tsdf-meshify",
        action="store_true",
        dest="skip_tsdf_meshify",
        help="Skip TSDF mesh generation and Chamfer distance evaluation",
    )
    parser.add_argument(
        "--tsdf-n-samples",
        type=int,
        default=10_000,
        dest="tsdf_n_samples",
        help="Number of surface samples for Chamfer distance (default: 10,000)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Summary CSV path (default: <output-dir>/train_eval_summary.csv)",
    )
    args = parser.parse_args()

    data_root = (
        (repo_root / args.data_dir).resolve()
        if not args.data_dir.is_absolute()
        else args.data_dir.resolve()
    )

    scenes = list_scenes(data_root)
    if args.models is not None:
        want = set(args.models)
        scenes = [s for s in scenes if s in want]
        missing = want - set(scenes)
        for m in sorted(missing):
            print(
                f"Warning: '{m}' not found or missing singleview_dynamic/gt/trans.json",
                file=sys.stderr,
            )

    if not scenes:
        print(
            f"No valid PartNetVideo scenes found under {data_root}.",
            file=sys.stderr,
        )
        return 1

    num_gpus = max(1, int(args.num_gpus))
    gpu_id = int(args.gpu_id)
    if gpu_id < 0 or gpu_id >= num_gpus:
        print(
            f"--gpu-id must satisfy 0 <= gpu-id < num-gpus (got gpu-id={gpu_id}, num-gpus={num_gpus}).",
            file=sys.stderr,
        )
        return 1

    all_scene_count = len(scenes)
    if num_gpus > 1:
        scenes = scenes[gpu_id::num_gpus]
        print(
            f"Sharding: gpu-id {gpu_id}/{num_gpus} → {len(scenes)} scene(s) "
            f"of {all_scene_count} total (stride slice [:: {num_gpus}] starting at {gpu_id}).",
            flush=True,
        )
        if not scenes:
            print(
                "No scenes in this shard; exiting successfully (check gpu-id vs dataset size).",
                flush=True,
            )
            return 0

    output_root = (
        args.output_dir.resolve()
        if args.output_dir.is_absolute()
        else (repo_root / args.output_dir).resolve()
    )
    if args.summary is not None:
        summary_path = (
            args.summary.resolve()
            if args.summary.is_absolute()
            else (repo_root / args.summary).resolve()
        )
    elif num_gpus > 1:
        summary_path = (
            output_root / f"train_eval_summary.shard{gpu_id}_of_{num_gpus}.csv"
        )
    else:
        summary_path = output_root / "train_eval_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not summary_path.is_file()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    failures: list[str] = []
    ts = datetime.now(timezone.utc).isoformat()

    for name in scenes:
        scene_dir  = data_root / name
        model_path = output_root / name
        gt_trans   = _gt_trans_path(scene_dir)

        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        # Parse trans.json once: needed both for the part-count cap and for training.
        try:
            num_parts, freeze_parts = _num_parts_from_trans(gt_trans)
        except Exception as exc:
            msg = f"{name}: could not parse trans.json: {exc}"
            print(msg, file=sys.stderr)
            _write_error(model_path, name, "parse_trans",
                         traceback.format_exc())
            failures.append(msg)
            continue

        # Skip scenes that exceed the part-count cap.
        if num_parts > args.max_parts:
            print(
                f"[SKIP] {name}: num_parts={num_parts} > --max-parts={args.max_parts}",
                flush=True,
            )
            continue

        # Completed scenes: skip entirely, re-eval only, or retrain (--force).
        final_ckpt = model_path / "ckpts" / f"ours_{args.iterations}.pth"
        training_complete = final_ckpt.is_file()

        if training_complete and not args.force and not args.reeval_completed:
            print(
                f"[SKIP] {name}: {final_ckpt.name} already exists "
                f"(use --reeval-completed to re-run eval, or --force to retrain)",
                flush=True,
            )
            continue

        skip_train_this = args.skip_train or (
            training_complete and args.reeval_completed and not args.force
        )
        if skip_train_this and training_complete and args.reeval_completed and not args.force:
            print(
                f"[SKIP TRAIN] {name}: {final_ckpt.name} exists; "
                f"re-running eval pipeline only",
                flush=True,
            )

        train_ok  = True
        eval_ok   = True
        render_ok = True
        dataset_render_ok = True
        structure_ok = True
        tsdf_ok = True
        chamfer_results = {}
        gaussian_count: int | None = None

        # ------------------------------------------------------------------
        # 1. TRAIN
        # ------------------------------------------------------------------
        if not skip_train_this:
            cmd = [
                sys.executable, str(repo_root / "train.py"),
                "-s", str(scene_dir),
                "-m", str(model_path),
                "-r", "1",
                "--eval",
                "--num_parts", str(num_parts),
                "--use_partnet_video",
                "--iterations", str(args.iterations),
                "--freeze_parts"
            ] + [str(i) for i in freeze_parts]

            print(f"[TRAIN] {' '.join(cmd)}")
            try:
                r = subprocess.run(cmd, cwd=str(repo_root), env=env)
                rc = r.returncode
            except Exception as exc:
                rc = -1
                print(f"[TRAIN] subprocess raised: {exc}", file=sys.stderr)
                _write_error(model_path, name, "train", traceback.format_exc())
            if rc != 0:
                train_ok = False
                msg = f"{name}: train.py exit {rc}"
                print(msg, file=sys.stderr)
                _write_error(model_path, name, "train",
                             f"Exit code: {rc}\nCmd: {' '.join(cmd)}")
                failures.append(msg)
                # record Gaussian count even on partial runs
                ply = _latest_ply(model_path)
                if ply:
                    gaussian_count = _count_gaussians(ply)

        # ------------------------------------------------------------------
        # 2. EVAL (before render: render_video.py reads results.json for iteration)
        #    Only runs if training succeeded (or was skipped).
        # ------------------------------------------------------------------
        if not args.skip_eval and train_ok:
            if not (model_path / "cfg_args").is_file():
                eval_ok = False
                msg = f"{name}: missing cfg_args (training may have failed)"
                print(msg, file=sys.stderr)
                _write_error(model_path, name, "eval", msg)
                failures.append(msg)
            else:
                cmd = [
                    sys.executable, str(repo_root / "eval_axis.py"),
                    "-m", str(model_path),
                    "--gt_path", str(gt_trans),
                ]
                print(f"[EVAL] {' '.join(cmd)}")
                try:
                    r = subprocess.run(cmd, cwd=str(repo_root), env=env)
                    rc = r.returncode
                except Exception as exc:
                    rc = -1
                    print(f"[EVAL] subprocess raised: {exc}", file=sys.stderr)
                    _write_error(model_path, name, "eval", traceback.format_exc())
                if rc != 0:
                    eval_ok = False
                    msg = f"{name}: eval_axis.py exit {rc}"
                    print(msg, file=sys.stderr)
                    _write_error(model_path, name, "eval",
                                 f"Exit code: {rc}\nCmd: {' '.join(cmd)}")
                    failures.append(msg)

        # ------------------------------------------------------------------
        # 2b. IMAGE METRICS (PSNR / SSIM / LPIPS on the test set)
        #     Only runs if training succeeded (or was skipped) and cfg_args exists.
        # ------------------------------------------------------------------
        image_metrics_ok = True
        if not args.skip_image_metrics and train_ok:
            metrics_script = repo_root / "eval_image_metrics.py"
            if not metrics_script.is_file():
                image_metrics_ok = False
                print(f"[METRICS] eval_image_metrics.py not found, skipping.")
            elif not (model_path / "cfg_args").is_file():
                image_metrics_ok = False
                msg = f"{name}: missing cfg_args, cannot run eval_image_metrics.py"
                print(msg, file=sys.stderr)
                _write_error(model_path, name, "image_metrics", msg)
                failures.append(msg)
            else:
                cmd = [
                    sys.executable, str(metrics_script),
                    "-m", str(model_path),
                ]
                print(f"[METRICS] {' '.join(cmd)}")
                try:
                    r = subprocess.run(cmd, cwd=str(repo_root), env=env)
                    rc = r.returncode
                except Exception as exc:
                    rc = -1
                    print(f"[METRICS] subprocess raised: {exc}", file=sys.stderr)
                    _write_error(model_path, name, "image_metrics",
                                 traceback.format_exc())
                if rc != 0:
                    image_metrics_ok = False
                    msg = f"{name}: eval_image_metrics.py exit {rc}"
                    print(msg, file=sys.stderr)
                    _write_error(model_path, name, "image_metrics",
                                 f"Exit code: {rc}\nCmd: {' '.join(cmd)}")
                    failures.append(msg)

        # ------------------------------------------------------------------
        # 2c. DATASET RENDER (train/test PNGs from dataset transforms)
        # ------------------------------------------------------------------
        if not args.skip_dataset_render and train_ok:
            views_render_script = repo_root / "render.py"
            if not views_render_script.is_file():
                dataset_render_ok = False
                print(f"[RENDER-VIEWS] render.py not found, skipping.")
            elif not (model_path / "cfg_args").is_file():
                dataset_render_ok = False
                msg = f"{name}: missing cfg_args, cannot run render.py"
                print(msg, file=sys.stderr)
                _write_error(model_path, name, "dataset_render", msg)
                failures.append(msg)
            else:
                cmd = [
                    sys.executable,
                    str(views_render_script),
                    "-m",
                    str(model_path),
                    "-s",
                    str(scene_dir),
                    "--multiview-dir",
                    args.multiview_dir,
                ]
                print(f"[RENDER-VIEWS] {' '.join(cmd)}")
                try:
                    r = subprocess.run(cmd, cwd=str(repo_root), env=env)
                    rc = r.returncode
                except Exception as exc:
                    rc = -1
                    print(f"[RENDER-VIEWS] subprocess raised: {exc}", file=sys.stderr)
                    _write_error(
                        model_path, name, "dataset_render", traceback.format_exc()
                    )
                if rc != 0:
                    dataset_render_ok = False
                    msg = f"{name}: render.py exit {rc}"
                    print(msg, file=sys.stderr)
                    _write_error(
                        model_path,
                        name,
                        "dataset_render",
                        f"Exit code: {rc}\nCmd: {' '.join(cmd)}",
                    )
                    failures.append(msg)

        # ------------------------------------------------------------------
        # 3. ORBITAL VIDEO (render_video.py)
        # ------------------------------------------------------------------
        if not args.skip_render and train_ok:
            render_script = repo_root / "render_video.py"
            if not render_script.is_file():
                render_ok = False
                print(f"[RENDER] render_video.py not found, skipping.")
            else:
                cmd = [
                    sys.executable, str(render_script),
                    "-m", str(model_path),
                ]
                print(f"[RENDER] {' '.join(cmd)}")
                try:
                    r = subprocess.run(cmd, cwd=str(repo_root), env=env)
                    rc = r.returncode
                except Exception as exc:
                    rc = -1
                    print(f"[RENDER] subprocess raised: {exc}", file=sys.stderr)
                    _write_error(model_path, name, "render", traceback.format_exc())
                if rc != 0:
                    render_ok = False
                    msg = f"{name}: render_video.py exit {rc}"
                    print(msg, file=sys.stderr)
                    _write_error(model_path, name, "render",
                                 f"Exit code: {rc}\nCmd: {' '.join(cmd)}")
                    failures.append(msg)

        # ------------------------------------------------------------------
        # 4. Ellipsoid structure video (orbit, Gaussian DC colors, no subsample)
        #    Only runs if training succeeded (or was skipped) and cameras.json exists.
        # ------------------------------------------------------------------
        if (
            not args.skip_structure_video
            and train_ok
            and (model_path / "cameras.json").is_file()
        ):
            vis_script = repo_root / "scripts" / "vis_gaussian_structure.py"
            if vis_script.is_file():
                cmd = [
                    sys.executable,
                    str(vis_script),
                    "-m",
                    str(model_path),
                    "--video",
                    "--trajectory",
                    "orbit",
                    "--use_gaussian_color",
                    "--max_gaussians",
                    "0",
                    "--out",
                    "gaussian_structure_orbit.mp4",
                ]
                print(f"[STRUCTURE] {' '.join(cmd)}")
                try:
                    r = subprocess.run(cmd, cwd=str(repo_root), env=env)
                    rc = r.returncode
                except Exception as exc:
                    rc = -1
                    print(f"[STRUCTURE] subprocess raised: {exc}", file=sys.stderr)
                    _write_error(
                        model_path, name, "structure_video", traceback.format_exc()
                    )
                if rc != 0:
                    structure_ok = False
                    msg = f"{name}: vis_gaussian_structure.py exit {rc}"
                    print(msg, file=sys.stderr)
                    _write_error(
                        model_path,
                        name,
                        "structure_video",
                        f"Exit code: {rc}\nCmd: {' '.join(cmd)}",
                    )
                    failures.append(msg)
            else:
                structure_ok = False
                msg = f"{name}: vis_gaussian_structure.py not found"
                print(f"[STRUCTURE] {msg}", file=sys.stderr)
                failures.append(msg)

        # ------------------------------------------------------------------
        # 5. TSDF mesh generation and Chamfer distance evaluation
        #    Requires training succeeded and cameras.json exists.
        # ------------------------------------------------------------------
        if (
            not args.skip_tsdf_meshify
            and train_ok
            and (model_path / "cameras.json").is_file()
        ):
            tsdf_script = repo_root / "tsdf_meshify.py"
            if not tsdf_script.is_file():
                tsdf_ok = False
                print(f"[TSDF] tsdf_meshify.py not found, skipping.")
            else:
                # Generate TSDF meshes for full, static, and dynamic parts
                meshes_dir = model_path / "meshes"
                
                for mode in ["full", "static", "dynamic"]:
                    cmd = [
                        sys.executable,
                        str(tsdf_script),
                        "--object_root",
                        str(model_path),
                        "--partnet_root",
                        str(data_root),
                        "--mode",
                        mode,
                    ]
                    print(f"[TSDF-{mode.upper()}] {' '.join(cmd)}")
                    try:
                        r = subprocess.run(cmd, cwd=str(repo_root), env=env)
                        rc = r.returncode
                    except Exception as exc:
                        rc = -1
                        print(f"[TSDF-{mode.upper()}] subprocess raised: {exc}", file=sys.stderr)
                        _write_error(
                            model_path, name, f"tsdf_{mode}", traceback.format_exc()
                        )
                    if rc != 0:
                        tsdf_ok = False
                        msg = f"{name}: tsdf_meshify.py ({mode}) exit {rc}"
                        print(msg, file=sys.stderr)
                        _write_error(
                            model_path,
                            name,
                            f"tsdf_{mode}",
                            f"Exit code: {rc}\nCmd: {' '.join(cmd)}",
                        )
                        failures.append(msg)
                
                # Compute Chamfer distances if TSDF generation succeeded
                if tsdf_ok:
                    print(f"[CHAMFER] Computing Chamfer distances for {name}...")
                    for mode in ["full", "static", "dynamic"]:
                        tsdf_mesh = meshes_dir / f"tsdf_mesh_{mode}.ply"
                        gt_mesh = scene_dir / "multiview_static" / f"gt_{mode}.ply"
                        
                        result = _compute_mesh_chamfer(
                            tsdf_mesh, gt_mesh, n_samples=args.tsdf_n_samples
                        )
                        if result is not None:
                            chamfer_results[f"chamfer_{mode}"] = result["chamfer_distance"]
                            print(
                                f"[CHAMFER] {mode:8s}: CD = {result['chamfer_distance']:.6f}"
                            )
                        else:
                            print(
                                f"[CHAMFER] {mode:8s}: skipped (mesh not found or error)",
                                file=sys.stderr,
                            )
                    
                    # Save Chamfer results to results.json
                    if chamfer_results:
                        save_results_json_merged(
                            model_path,
                            {"chamfer_metrics": chamfer_results},
                        )

        # ------------------------------------------------------------------
        # 6. Record Gaussian count in results.json
        # ------------------------------------------------------------------
        ply = _latest_ply(model_path)
        if ply is not None:
            gaussian_count = _count_gaussians(ply)
            if gaussian_count is not None:
                save_results_json_merged(
                    model_path,
                    {
                        "point_cloud_stats": {
                            "gaussian_count": gaussian_count,
                            "ply_path": str(ply),
                        },
                    },
                )
                print(f"[INFO] Gaussian primitives: {gaussian_count}")

        # ------------------------------------------------------------------
        # 7. CSV summary row
        # ------------------------------------------------------------------
        rjson = results_json_path(model_path)
        if rjson.is_file():
            try:
                results_rel = str(rjson.relative_to(repo_root))
            except ValueError:
                results_rel = str(rjson)
        else:
            results_rel = ""
        with summary_path.open("a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow([
                    "timestamp_utc", "scene",
                    "train_ok", "render_ok", "dataset_render_ok", "eval_ok",
                    "image_metrics_ok", "structure_ok", "tsdf_ok",
                    "gaussian_count", "chamfer_full", "chamfer_static", "chamfer_dynamic",
                    "results_json",
                ])
                write_header = False
            if skip_train_this:
                train_csv = (
                    "skipped_complete"
                    if training_complete and args.reeval_completed and not args.force
                    else "skipped"
                )
            else:
                train_csv = str(train_ok)
            w.writerow([
                ts, name,
                train_csv,
                str(render_ok if not args.skip_render else "skipped"),
                str(dataset_render_ok if not args.skip_dataset_render else "skipped"),
                str(eval_ok   if not args.skip_eval   else "skipped"),
                str(image_metrics_ok if not args.skip_image_metrics else "skipped"),
                str(structure_ok if not args.skip_structure_video else "skipped"),
                str(tsdf_ok if not args.skip_tsdf_meshify else "skipped"),
                str(gaussian_count) if gaussian_count is not None else "",
                f"{chamfer_results['chamfer_full']:.6f}" if 'chamfer_full' in chamfer_results else "",
                f"{chamfer_results['chamfer_static']:.6f}" if 'chamfer_static' in chamfer_results else "",
                f"{chamfer_results['chamfer_dynamic']:.6f}" if 'chamfer_dynamic' in chamfer_results else "",
                results_rel,
            ])

    if failures:
        if num_gpus > 1:
            fail_log = (
                summary_path.parent / f"failures.shard{gpu_id}_of_{num_gpus}.txt"
            )
        else:
            fail_log = summary_path.parent / "failures.txt"
        with fail_log.open("w") as f:
            f.write(f"Run: {ts}\n\n")
            for msg in failures:
                f.write(msg + "\n")
        print(
            f"\nCompleted with {len(failures)} error(s). See {fail_log}",
            file=sys.stderr,
        )
        return 1

    print(f"\nAll done. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
