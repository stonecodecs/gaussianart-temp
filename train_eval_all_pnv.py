#!/usr/bin/env python3
"""
Train, render, and evaluate every scene in a PartNet-Video (data120-style) dataset.

Dataset layout expected per scene::

    <data_root>/<scene_name>/
        multiview_static_start/   ← start-state cameras
        multiview_static/         ← end-state cameras
        multiview_static/gt/trans.json   ← joint ground-truth
        points3d.ply
        semantics.npy

Outputs per scene::

    <output_dir>/<scene_name>/
        ckpts/ours_*.pth           ← articulation checkpoints (from train.py)
        point_cloud/               ← Gaussian splats
        results.txt                ← eval_axis metrics + Gaussian count
        error.txt                  ← only written when a stage fails

A CSV summary is appended to ``<output_dir>/train_eval_summary.csv``.

Example::

    python train_eval_all_pnv.py --data-dir data120 --gpu 0
    python train_eval_all_pnv.py --data-dir data120 --models Safe_101605 --skip-train
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gt_trans_path(scene_dir: Path) -> Path | None:
    """Return the trans.json path for a PartNet-Video scene, or None."""
    candidate = scene_dir / "multiview_static" / "gt" / "trans.json"
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
        type=str,
        default="output/PartNetVideo",
        help="Training output directory (relative to repo root)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Only process these scene names (default: all valid scenes)",
    )
    parser.add_argument("--skip-train",  action="store_true", help="Skip training")
    parser.add_argument("--skip-eval",   action="store_true", help="Skip evaluation")
    parser.add_argument("--skip-render", action="store_true", help="Skip render_video step")
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
                f"Warning: '{m}' not found or missing multiview_static/gt/trans.json",
                file=sys.stderr,
            )

    if not scenes:
        print(
            f"No valid PartNetVideo scenes found under {data_root}.",
            file=sys.stderr,
        )
        return 1

    out_rel = args.output_dir.strip("/")
    summary_path = (
        args.summary.resolve()
        if args.summary is not None and args.summary.is_absolute()
        else (
            args.summary.resolve()
            if args.summary is not None
            else (repo_root / out_rel / "train_eval_summary.csv").resolve()
        )
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not summary_path.is_file()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    failures: list[str] = []
    ts = datetime.now(timezone.utc).isoformat()

    for name in scenes:
        scene_dir  = data_root / name
        model_path = repo_root / out_rel / name
        gt_trans   = _gt_trans_path(scene_dir)

        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        train_ok  = True
        eval_ok   = True
        render_ok = True
        gaussian_count: int | None = None

        # ------------------------------------------------------------------
        # 1. TRAIN
        # ------------------------------------------------------------------
        if not args.skip_train:
            try:
                num_parts, freeze_parts = _num_parts_from_trans(gt_trans)
            except Exception as exc:
                msg = f"{name}: could not parse trans.json: {exc}"
                print(msg, file=sys.stderr)
                _write_error(model_path, name, "parse_trans",
                             traceback.format_exc())
                failures.append(msg)
                continue

            cmd = [
                sys.executable, str(repo_root / "train.py"),
                "-s", str(scene_dir),
                "-m", str(model_path),
                "-r", "1",
                "--eval",
                "--num_parts", str(num_parts),
                "--use_partnet_video",
                "--freeze_parts",
            ] + [str(i) for i in freeze_parts]

            print(f"[TRAIN] {' '.join(cmd)}")
            r = subprocess.run(cmd, cwd=str(repo_root), env=env)
            if r.returncode != 0:
                train_ok = False
                msg = f"{name}: train.py exit {r.returncode}"
                print(msg, file=sys.stderr)
                _write_error(model_path, name, "train",
                             f"Exit code: {r.returncode}\nCmd: {' '.join(cmd)}")
                failures.append(msg)
                # record Gaussian count even on partial runs
                ply = _latest_ply(model_path)
                if ply:
                    gaussian_count = _count_gaussians(ply)

        # ------------------------------------------------------------------
        # 2. RENDER
        # ------------------------------------------------------------------
        if not args.skip_render and train_ok:
            render_script = repo_root / "render_video.py"
            if render_script.is_file():
                cmd = [
                    sys.executable, str(render_script),
                    "-m", str(model_path),
                ]
                print(f"[RENDER] {' '.join(cmd)}")
                r = subprocess.run(cmd, cwd=str(repo_root), env=env)
                if r.returncode != 0:
                    render_ok = False
                    msg = f"{name}: render_video.py exit {r.returncode}"
                    print(msg, file=sys.stderr)
                    _write_error(model_path, name, "render",
                                 f"Exit code: {r.returncode}\nCmd: {' '.join(cmd)}")
                    failures.append(msg)
            else:
                print(f"[RENDER] render_video.py not found, skipping.")
                render_ok = False

        # ------------------------------------------------------------------
        # 3. EVAL
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
                r = subprocess.run(cmd, cwd=str(repo_root), env=env)
                if r.returncode != 0:
                    eval_ok = False
                    msg = f"{name}: eval_axis.py exit {r.returncode}"
                    print(msg, file=sys.stderr)
                    _write_error(model_path, name, "eval",
                                 f"Exit code: {r.returncode}\nCmd: {' '.join(cmd)}")
                    failures.append(msg)

        # ------------------------------------------------------------------
        # 4. Record Gaussian count into results.txt
        # ------------------------------------------------------------------
        ply = _latest_ply(model_path)
        if ply is not None:
            gaussian_count = _count_gaussians(ply)
            if gaussian_count is not None:
                results_file = model_path / "results.txt"
                # Append if file already exists; write fresh otherwise
                mode = "a" if results_file.is_file() else "w"
                with results_file.open(mode) as f:
                    f.write(f"Gaussian count: {gaussian_count}\n")
                    f.write(f"PLY: {ply}\n")
                print(f"[INFO] Gaussian primitives: {gaussian_count}")

        # ------------------------------------------------------------------
        # 5. CSV summary row
        # ------------------------------------------------------------------
        results_rel = (
            os.path.relpath(model_path / "results.txt", repo_root)
            if (model_path / "results.txt").is_file()
            else ""
        )
        with summary_path.open("a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow([
                    "timestamp_utc", "scene",
                    "train_ok", "render_ok", "eval_ok",
                    "gaussian_count", "results_txt",
                ])
                write_header = False
            w.writerow([
                ts, name,
                str(train_ok  if not args.skip_train  else "skipped"),
                str(render_ok if not args.skip_render else "skipped"),
                str(eval_ok   if not args.skip_eval   else "skipped"),
                str(gaussian_count) if gaussian_count is not None else "",
                results_rel,
            ])

    if failures:
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
