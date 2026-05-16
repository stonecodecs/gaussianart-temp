#!/usr/bin/env python3
"""
Train and evaluate every scene under ./data (each folder with gt/trans.json).

Uses the same output layout as run.py: output/MPArt90/<scene_name>.

Example:
  python train_eval_all.py --gpu 0
  python train_eval_all.py --models Knife_101115 Door_8961 --skip-eval
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def list_scenes(data_root: Path) -> list[str]:
    if not data_root.is_dir():
        return []
    scenes: list[str] = []
    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "gt" / "trans.json").is_file():
            scenes.append(child.name)
    return scenes


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run run.py then eval_axis.py for all scenes in ./data"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "data",
        help="Dataset root containing scene folders (default: ./data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/MPArt90",
        help="Training output directory, relative to repo root (default: output/MPArt90)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Only these scene folder names (default: all valid scenes)",
    )
    parser.add_argument("--skip-train", action="store_true", help="Only run evaluation")
    parser.add_argument("--skip-eval", action="store_true", help="Only run training")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep going after a failed scene",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Append per-scene summary CSV (default: <output-dir>/train_eval_summary.csv)",
    )
    args = parser.parse_args()

    data_root = (repo_root / args.data_dir).resolve() if not args.data_dir.is_absolute() else args.data_dir.resolve()

    scenes = list_scenes(data_root)
    if args.models is not None:
        want = set(args.models)
        scenes = [s for s in scenes if s in want]
        missing = want - set(scenes)
        for m in sorted(missing):
            print(
                f"Warning: requested scene not found or missing gt/trans.json: {m}",
                file=sys.stderr,
            )

    if not scenes:
        print(f"No scenes found under {data_root} (need <scene>/gt/trans.json).", file=sys.stderr)
        return 1

    out_rel = args.output_dir.strip("/").strip()
    if args.summary is None:
        summary_path = (repo_root / out_rel / "train_eval_summary.csv").resolve()
    else:
        summary_path = (
            args.summary.resolve()
            if args.summary.is_absolute()
            else (repo_root / args.summary).resolve()
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_path.is_file()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    failures: list[str] = []
    ts = datetime.now(timezone.utc).isoformat()

    for name in scenes:
        model_path = repo_root / out_rel / name
        print(f"\n========== {name} ==========")

        train_ok = True
        eval_ok = True

        if not args.skip_train:
            r = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "run.py"),
                    "--model_id",
                    name,
                    "--root_dir",
                    str(data_root),
                    "--gpu",
                    str(args.gpu),
                ],
                cwd=str(repo_root),
                env=env,
            )
            if r.returncode != 0:
                train_ok = False
                msg = f"{name}: run.py exit {r.returncode}"
                print(msg, file=sys.stderr)
                failures.append(msg)
                if not args.continue_on_error:
                    return r.returncode

        if not args.skip_eval:
            if not train_ok:
                eval_ok = False
            elif not (model_path / "cfg_args").is_file():
                eval_ok = False
                msg = f"{name}: missing {model_path / 'cfg_args'} (train failed or wrong --output-dir)"
                print(msg, file=sys.stderr)
                failures.append(msg)
                if not args.continue_on_error:
                    return 1
            else:
                r = subprocess.run(
                    [sys.executable, str(repo_root / "eval_axis.py"), "-m", str(model_path)],
                    cwd=str(repo_root),
                    env=env,
                )
                if r.returncode != 0:
                    eval_ok = False
                    msg = f"{name}: eval_axis.py exit {r.returncode}"
                    print(msg, file=sys.stderr)
                    failures.append(msg)
                    if not args.continue_on_error:
                        return r.returncode

        results_rel = ""
        results_json = model_path / "results.json"
        if results_json.is_file():
            results_rel = os.path.relpath(results_json, repo_root)

        with summary_path.open("a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(
                    [
                        "timestamp_utc",
                        "scene",
                        "train_ok",
                        "eval_ok",
                        "results_json",
                    ]
                )
                write_header = False
            w.writerow(
                [
                    ts,
                    name,
                    str(train_ok if not args.skip_train else "skipped"),
                    str(eval_ok if not args.skip_eval else "skipped"),
                    results_rel,
                ]
            )

    if failures:
        print(f"\nCompleted with {len(failures)} error(s).", file=sys.stderr)
        return 1
    print(f"\nAll done. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
