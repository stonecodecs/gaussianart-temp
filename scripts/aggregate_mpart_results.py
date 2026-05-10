#!/usr/bin/env python3
"""
Aggregate eval metrics from GaussianArt/MPArt `results.txt` files under an output tree.

Valid files match eval_axis.py output: Evaluated iteration (or legacy The best), Parts num,
Angle mean, Distance mean, Theta diff mean, and optional Part N: lines. Skips dirs where
results.txt is missing or incomplete.

Usage:
  python scripts/aggregate_mpart_results.py --root output/MPArt90
  python scripts/aggregate_mpart_results.py --root output/MPArt90 --json summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SceneResult:
    name: str
    parts_num: int
    angle_mean: float
    distance_mean: float
    theta_diff_mean: float
    part_rows: list[tuple[float, float, float]] = field(default_factory=list)


PART_LINE = re.compile(
    r"^Part\s+(\d+):\s*angle=([\d.eE+-]+),\s*distance=([\d.eE+-]+),\s*theta_diff=([\d.eE+-]+)\s*$"
)


def parse_results_txt(path: Path) -> SceneResult | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None

    parts_num: int | None = None
    angle_mean = distance_mean = theta_diff_mean = None
    part_rows: list[tuple[float, float, float]] = []

    for ln in lines:
        if ln.startswith("Parts num:"):
            try:
                parts_num = int(ln.split(":", 1)[1].strip())
            except ValueError:
                return None
        elif ln.startswith("Angle mean:"):
            try:
                angle_mean = float(ln.split(":", 1)[1].strip())
            except ValueError:
                return None
        elif ln.startswith("Distance mean:"):
            try:
                distance_mean = float(ln.split(":", 1)[1].strip())
            except ValueError:
                return None
        elif ln.startswith("Theta diff mean:"):
            try:
                theta_diff_mean = float(ln.split(":", 1)[1].strip())
            except ValueError:
                return None
        else:
            m = PART_LINE.match(ln)
            if m:
                part_rows.append(
                    (float(m.group(2)), float(m.group(3)), float(m.group(4)))
                )

    if parts_num is None or angle_mean is None or distance_mean is None or theta_diff_mean is None:
        return None

    name = path.parent.name
    return SceneResult(
        name=name,
        parts_num=parts_num,
        angle_mean=angle_mean,
        distance_mean=distance_mean,
        theta_diff_mean=theta_diff_mean,
        part_rows=part_rows,
    )


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.fmean(xs), statistics.stdev(xs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate results.txt metrics under a root folder")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output/MPArt90"),
        help="Directory containing per-scene subfolders with results.txt",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to write machine-readable summary JSON",
    )
    args = parser.parse_args()

    root: Path = args.root.resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    all_paths = sorted(root.glob("**/results.txt"))
    parsed: list[SceneResult] = []
    skipped: list[tuple[Path, str]] = []

    for p in all_paths:
        r = parse_results_txt(p)
        if r is None:
            skipped.append((p, "parse failed or incomplete"))
            continue
        parsed.append(r)

    n = len(parsed)
    if n == 0:
        print(f"No valid results.txt under {root}")
        sys.exit(0)

    # --- Global averages (scene-level means) ---
    am = [s.angle_mean for s in parsed]
    dm = [s.distance_mean for s in parsed]
    tm = [s.theta_diff_mean for s in parsed]
    g_am, g_am_sd = mean_std(am)
    g_dm, g_dm_sd = mean_std(dm)
    g_tm, g_tm_sd = mean_std(tm)

    # --- By parts_num: scene-level averages ---
    by_parts: dict[int, list[SceneResult]] = {}
    for s in parsed:
        by_parts.setdefault(s.parts_num, []).append(s)

    by_k_summary: dict[str, dict] = {}
    for k in sorted(by_parts.keys()):
        subs = by_parts[k]
        sa = [x.angle_mean for x in subs]
        sd = [x.distance_mean for x in subs]
        st = [x.theta_diff_mean for x in subs]
        # Pooled per-part metrics (all Part lines from scenes with this part count)
        pooled_a: list[float] = []
        pooled_d: list[float] = []
        pooled_t: list[float] = []
        for x in subs:
            for a, d, t in x.part_rows:
                pooled_a.append(a)
                pooled_d.append(d)
                pooled_t.append(t)
        pa_m, pa_sd = mean_std(pooled_a)
        pd_m, pd_sd = mean_std(pooled_d)
        pt_m, pt_sd = mean_std(pooled_t)
        by_k_summary[str(k)] = {
            "num_scenes": len(subs),
            "scene_level": {
                "angle_mean_avg": statistics.fmean(sa),
                "angle_mean_std": mean_std(sa)[1],
                "distance_mean_avg": statistics.fmean(sd),
                "distance_mean_std": mean_std(sd)[1],
                "theta_diff_mean_avg": statistics.fmean(st),
                "theta_diff_mean_std": mean_std(st)[1],
            },
            "pooled_part_rows": {
                "count_part_lines": len(pooled_a),
                "angle_avg": pa_m,
                "angle_std": pa_sd,
                "distance_avg": pd_m,
                "distance_std": pd_sd,
                "theta_diff_avg": pt_m,
                "theta_diff_std": pt_sd,
            },
        }

    out = {
        "root": str(root),
        "valid_scenes": n,
        "skipped_files": len(skipped),
        "global_scene_level": {
            "angle_mean_avg": g_am,
            "angle_mean_std": g_am_sd,
            "distance_mean_avg": g_dm,
            "distance_mean_std": g_dm_sd,
            "theta_diff_mean_avg": g_tm,
            "theta_diff_mean_std": g_tm_sd,
        },
        "by_parts_num": by_k_summary,
        "scenes": [
            {
                "name": s.name,
                "parts_num": s.parts_num,
                "angle_mean": s.angle_mean,
                "distance_mean": s.distance_mean,
                "theta_diff_mean": s.theta_diff_mean,
                "num_part_lines": len(s.part_rows),
            }
            for s in sorted(parsed, key=lambda x: x.name)
        ],
    }

    # --- Human-readable report ---
    print(f"Root: {root}")
    print(f"Valid results.txt: {n}  |  Skipped (invalid/incomplete): {len(skipped)}")
    print()
    print("Global average (mean of each scene’s Angle mean / Distance mean / Theta diff mean):")
    print(f"  angle_mean:        {g_am:.6f}  (stdev across scenes: {g_am_sd:.6f})")
    print(f"  distance_mean:     {g_dm:.6f}  (stdev across scenes: {g_dm_sd:.6f})")
    print(f"  theta_diff_mean:   {g_tm:.6f}  (stdev across scenes: {g_tm_sd:.6f})")
    print()
    print("By Parts num (N):")
    for k in sorted(by_parts.keys()):
        subs = by_parts[k]
        sa = statistics.fmean([x.angle_mean for x in subs])
        sd = statistics.fmean([x.distance_mean for x in subs])
        st = statistics.fmean([x.theta_diff_mean for x in subs])
        pooled_n = sum(len(x.part_rows) for x in subs)
        print(f"  N={k}:  {len(subs)} scenes")
        print(f"    Scene-level avg:  angle_mean={sa:.6f}  distance_mean={sd:.6f}  theta_diff_mean={st:.6f}")
        if pooled_n:
            pa = statistics.fmean([a for x in subs for a, _, _ in x.part_rows])
            pd = statistics.fmean([d for x in subs for _, d, _ in x.part_rows])
            pt = statistics.fmean([t for x in subs for _, _, t in x.part_rows])
            print(
                f"    Pooled Part lines ({pooled_n} rows):  angle={pa:.6f}  distance={pd:.6f}  theta_diff={pt:.6f}"
            )
        else:
            print("    (no per-part lines parsed)")
    print()

    if skipped and len(skipped) <= 30:
        print("Skipped:")
        for p, _ in skipped[:30]:
            print(f"  {p.relative_to(root)}")
    elif skipped:
        print(f"Skipped {len(skipped)} files (omit listing; use --json for full list)")

    if args.json:
        # include skipped paths in json for debugging
        out["skipped"] = [str(p.relative_to(root)) for p, _ in skipped]
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
