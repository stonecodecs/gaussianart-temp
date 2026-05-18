#!/usr/bin/env python3
"""
Generate LaTeX table aggregating metrics from GaussianArt results.json files.

Reads results.json files containing axis_eval, image_metrics, and chamfer_metrics.
Aggregates by number of parts (2, 3, 4-5, 6+, All) and generates a formatted LaTeX table.

Usage:
  python scripts/get_latex_table.py --root /path/to/results --output table.tex
  python scripts/get_latex_table.py --root /path/to/results --stage start
  python scripts/get_latex_table.py --root /path/to/results --stage end
  python scripts/get_latex_table.py --root /path/to/results --stage both
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class SceneMetrics:
    name: str
    parts_num: int
    # Axis metrics
    angle_mean_deg: float
    distance_mean: float
    theta_diff_mean: float
    # Image metrics
    psnr_start: float | None
    ssim_start: float | None
    lpips_start: float | None
    psnr_end: float | None
    ssim_end: float | None
    lpips_end: float | None
    psnr_mean: float | None
    ssim_mean: float | None
    lpips_mean: float | None
    # Chamfer metrics
    chamfer_full: float | None
    chamfer_static: float | None
    chamfer_dynamic: float | None


def parse_results_json(path: Path) -> SceneMetrics | None:
    """Parse a results.json file and extract all metrics."""
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    # Parse axis_eval
    ae = data.get("axis_eval")
    if not isinstance(ae, dict):
        return None

    try:
        parts_num = int(ae["parts_num"])
        angle_mean_deg = float(ae.get("angle_mean_deg", ae.get("angle_mean")))
        distance_mean = float(ae["distance_mean"])
        theta_diff_mean = float(ae["theta_diff_mean"])
    except (KeyError, TypeError, ValueError):
        return None

    # Parse image_metrics
    im = data.get("image_metrics", {})
    psnr_start = None
    ssim_start = None
    lpips_start = None
    psnr_end = None
    ssim_end = None
    lpips_end = None
    psnr_mean = None
    ssim_mean = None
    lpips_mean = None

    if isinstance(im, dict):
        try:
            psnr_mean = float(im.get("psnr_mean", float("nan")))
            ssim_mean = float(im.get("ssim_mean", float("nan")))
            lpips_mean = float(im.get("lpips_mean", float("nan")))
            
            start_dict = im.get("start", {})
            if isinstance(start_dict, dict):
                psnr_start = float(start_dict.get("psnr", float("nan")))
                ssim_start = float(start_dict.get("ssim", float("nan")))
                lpips_start = float(start_dict.get("lpips", float("nan")))
            
            end_dict = im.get("end", {})
            if isinstance(end_dict, dict):
                psnr_end = float(end_dict.get("psnr", float("nan")))
                ssim_end = float(end_dict.get("ssim", float("nan")))
                lpips_end = float(end_dict.get("lpips", float("nan")))
        except (TypeError, ValueError):
            pass

    # Parse chamfer_metrics
    cm = data.get("chamfer_metrics", {})
    chamfer_full = None
    chamfer_static = None
    chamfer_dynamic = None

    if isinstance(cm, dict):
        try:
            chamfer_full = float(cm.get("chamfer_full", float("nan")))
            chamfer_static = float(cm.get("chamfer_static", float("nan")))
            chamfer_dynamic = float(cm.get("chamfer_dynamic", float("nan")))
        except (TypeError, ValueError):
            pass

    return SceneMetrics(
        name=path.parent.name,
        parts_num=parts_num,
        angle_mean_deg=angle_mean_deg,
        distance_mean=distance_mean,
        theta_diff_mean=theta_diff_mean,
        psnr_start=psnr_start,
        ssim_start=ssim_start,
        lpips_start=lpips_start,
        psnr_end=psnr_end,
        ssim_end=ssim_end,
        lpips_end=lpips_end,
        psnr_mean=psnr_mean,
        ssim_mean=ssim_mean,
        lpips_mean=lpips_mean,
        chamfer_full=chamfer_full,
        chamfer_static=chamfer_static,
        chamfer_dynamic=chamfer_dynamic,
    )


@dataclass
class AggregatedMetrics:
    """Aggregated metrics for a group of scenes."""
    num_scenes: int
    # Axis metrics
    axis_ang: float
    axis_pos: float
    part_motion: float
    # Chamfer metrics
    cd_s: float
    cd_m: float
    cd_w: float
    # Image metrics (depending on stage)
    psnr: float
    ssim: float
    lpips: float


def safe_mean(values: list[float | None]) -> float:
    """Compute mean, filtering out None and NaN values."""
    valid = [v for v in values if v is not None and not (isinstance(v, float) and v != v)]
    if not valid:
        return float("nan")
    return statistics.fmean(valid)


def aggregate_metrics(scenes: list[SceneMetrics], stage: Literal["start", "end", "both", "mean"]) -> AggregatedMetrics:
    """Aggregate metrics from a list of scenes."""
    if not scenes:
        return AggregatedMetrics(
            num_scenes=0,
            axis_ang=float("nan"),
            axis_pos=float("nan"),
            part_motion=float("nan"),
            cd_s=float("nan"),
            cd_m=float("nan"),
            cd_w=float("nan"),
            psnr=float("nan"),
            ssim=float("nan"),
            lpips=float("nan"),
        )

    # Aggregate axis metrics
    axis_ang = safe_mean([s.angle_mean_deg for s in scenes])
    axis_pos = safe_mean([s.distance_mean for s in scenes])
    part_motion = safe_mean([s.theta_diff_mean for s in scenes])

    # Aggregate chamfer metrics
    cd_s = safe_mean([s.chamfer_static for s in scenes])
    cd_m = safe_mean([s.chamfer_dynamic for s in scenes])
    cd_w = safe_mean([s.chamfer_full for s in scenes])

    # Aggregate image metrics based on stage
    if stage == "start":
        psnr = safe_mean([s.psnr_start for s in scenes])
        ssim = safe_mean([s.ssim_start for s in scenes])
        lpips = safe_mean([s.lpips_start for s in scenes])
    elif stage == "end":
        psnr = safe_mean([s.psnr_end for s in scenes])
        ssim = safe_mean([s.ssim_end for s in scenes])
        lpips = safe_mean([s.lpips_end for s in scenes])
    elif stage == "mean":
        psnr = safe_mean([s.psnr_mean for s in scenes])
        ssim = safe_mean([s.ssim_mean for s in scenes])
        lpips = safe_mean([s.lpips_mean for s in scenes])
    else:  # both
        # Average start and end
        psnr_vals = []
        ssim_vals = []
        lpips_vals = []
        for s in scenes:
            if s.psnr_start is not None:
                psnr_vals.append(s.psnr_start)
            if s.psnr_end is not None:
                psnr_vals.append(s.psnr_end)
            if s.ssim_start is not None:
                ssim_vals.append(s.ssim_start)
            if s.ssim_end is not None:
                ssim_vals.append(s.ssim_end)
            if s.lpips_start is not None:
                lpips_vals.append(s.lpips_start)
            if s.lpips_end is not None:
                lpips_vals.append(s.lpips_end)
        psnr = safe_mean(psnr_vals)
        ssim = safe_mean(ssim_vals)
        lpips = safe_mean(lpips_vals)

    return AggregatedMetrics(
        num_scenes=len(scenes),
        axis_ang=axis_ang,
        axis_pos=axis_pos,
        part_motion=part_motion,
        cd_s=cd_s,
        cd_m=cd_m,
        cd_w=cd_w,
        psnr=psnr,
        ssim=ssim,
        lpips=lpips,
    )


def format_metric(value: float, precision: int = 2, scale: float = 1.0) -> str:
    """Format a metric value for LaTeX, handling NaN."""
    if value != value:  # NaN check
        return "--"
    return f"{value * scale:.{precision}f}"


def find_best_and_second(values: list[tuple[str, float]], higher_is_better: bool) -> tuple[set[str], set[str]]:
    """Find best and second-best values, handling ties."""
    valid = [(name, val) for name, val in values if val == val]  # Filter NaN
    if not valid:
        return set(), set()
    
    sorted_vals = sorted(valid, key=lambda x: x[1], reverse=higher_is_better)
    
    if len(sorted_vals) == 0:
        return set(), set()
    
    best_val = sorted_vals[0][1]
    best = {name for name, val in sorted_vals if val == best_val}
    
    remaining = [x for x in sorted_vals if x[1] != best_val]
    if not remaining:
        return best, set()
    
    second_val = remaining[0][1]
    second = {name for name, val in remaining if val == second_val}
    
    return best, second


def format_cell(value: float, is_best: bool, is_second: bool, precision: int = 2, scale: float = 1.0) -> str:
    """Format a table cell with appropriate highlighting."""
    formatted = format_metric(value, precision, scale)
    if formatted == "--":
        return formatted
    
    if is_best:
        return f"\\cellcolor{{best}}\\textbf{{{formatted}}}"
    elif is_second:
        return f"\\cellcolor{{better}}{formatted}"
    else:
        return formatted


def group_column_label(group: str, metrics_by_group: dict[str, AggregatedMetrics]) -> str:
    """Format a part-count group label for a table column header, with object count."""
    n = metrics_by_group[group].num_scenes
    if group == "All":
        return f"All ({n})"
    return f"{group} parts ({n})"


def generate_latex_table(
    metrics_by_group: dict[str, AggregatedMetrics],
    stage: Literal["start", "end", "both", "mean"],
    output_path: Path | None = None
) -> str:
    """Generate LaTeX table from aggregated metrics (metrics as rows, part groups as columns)."""
    group_order = ["2", "3", "4-5", "6+", "All"]
    groups = [g for g in group_order if g in metrics_by_group]
    if not groups:
        return ""

    # (row label, arrow, getter, precision, scale, higher_is_better)
    metric_rows: list[tuple[str, str, object, int, float, bool]] = [
        ("Axis Ang", r"$\downarrow$", lambda m: m.axis_ang, 2, 1.0, False),
        ("Axis Pos", r"$\downarrow$", lambda m: m.axis_pos, 4, 1.0, False),
        ("Part Motion", r"$\downarrow$", lambda m: m.part_motion, 4, 1.0, False),
        ("CD-s", r"$\downarrow$", lambda m: m.cd_s, 4, 1000.0, False),
        ("CD-m", r"$\downarrow$", lambda m: m.cd_m, 4, 1000.0, False),
        ("CD-w", r"$\downarrow$", lambda m: m.cd_w, 4, 1000.0, False),
        ("PSNR", r"$\uparrow$", lambda m: m.psnr, 2, 1.0, True),
        ("SSIM", r"$\uparrow$", lambda m: m.ssim, 4, 1.0, True),
        ("LPIPS", r"$\downarrow$", lambda m: m.lpips, 4, 1.0, False),
    ]

    stage_label = {"start": "Start", "end": "End", "both": "Start+End", "mean": "Mean"}[stage]
    col_spec = "l" + "c" * len(groups)

    latex = []
    latex.append("\\begin{table}[ht]")
    latex.append("\\centering")
    latex.append(f"\\caption{{Quantitative results aggregated by number of parts ({stage_label} stage).\\\\")
    latex.append("$\\uparrow$: higher is better, $\\downarrow$: lower is better. \\colorbox{best}{Best} and \\colorbox{better}{2nd best} results are colored accordingly.}")
    latex.append("\\vspace{0.3em}")
    latex.append("")
    latex.append("\\adjustbox{max width=\\linewidth}{")
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\toprule")
    header_cols = " & ".join(
        f"\\textbf{{{group_column_label(g, metrics_by_group)}}}" for g in groups
    )
    latex.append(f"\\textbf{{Metric}} & {header_cols} \\\\")
    latex.append("\\midrule")

    for label, arrow, getter, precision, scale, higher_is_better in metric_rows:
        row_vals = [(g, getter(metrics_by_group[g])) for g in groups]
        best, second = find_best_and_second(row_vals, higher_is_better)
        cells = [
            format_cell(val, g in best, g in second, precision, scale)
            for g, val in row_vals
        ]
        latex.append(f"\\textbf{{{label}}} {arrow} & " + " & ".join(cells) + " \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("}")
    latex.append("\\end{table}")
    
    result = "\n".join(latex)
    
    # Write to file if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result, encoding="utf-8")
        print(f"LaTeX table written to: {output_path}")
    
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX table from GaussianArt results.json files"
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing per-scene subfolders with results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for LaTeX table (default: print to stdout)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["start", "end", "both", "mean"],
        default="mean",
        help="Which stage to use for image metrics: 'start', 'end', 'both' (average both), or 'mean' (use psnr_mean/ssim_mean/lpips_mean)",
    )
    args = parser.parse_args()

    root: Path = args.root.resolve()
    if not root.is_dir():
        print(f"Error: Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    # Find all results.json files
    all_paths = sorted(root.glob("**/results.json"))
    parsed: list[SceneMetrics] = []
    skipped = 0

    for p in all_paths:
        r = parse_results_json(p)
        if r is None:
            skipped += 1
            continue
        parsed.append(r)

    n = len(parsed)
    if n == 0:
        print(f"Error: No valid results.json found under {root}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {n} valid results.json files (skipped {skipped} invalid)")
    print()

    # Group scenes by parts_num
    by_parts: dict[int, list[SceneMetrics]] = {}
    for s in parsed:
        by_parts.setdefault(s.parts_num, []).append(s)

    # Create aggregated groups
    metrics_by_group: dict[str, AggregatedMetrics] = {}
    
    # 2 parts
    if 2 in by_parts:
        metrics_by_group["2"] = aggregate_metrics(by_parts[2], args.stage)
    
    # 3 parts
    if 3 in by_parts:
        metrics_by_group["3"] = aggregate_metrics(by_parts[3], args.stage)
    
    # 4-5 parts
    scenes_4_5 = []
    for k in [4, 5]:
        scenes_4_5.extend(by_parts.get(k, []))
    if scenes_4_5:
        metrics_by_group["4-5"] = aggregate_metrics(scenes_4_5, args.stage)
    
    # 6+ parts
    scenes_6plus = []
    for k in by_parts.keys():
        if k >= 6:
            scenes_6plus.extend(by_parts[k])
    if scenes_6plus:
        metrics_by_group["6+"] = aggregate_metrics(scenes_6plus, args.stage)
    
    # All
    metrics_by_group["All"] = aggregate_metrics(parsed, args.stage)

    # Print summary
    print("Summary by group:")
    for group in ["2", "3", "4-5", "6+", "All"]:
        if group in metrics_by_group:
            m = metrics_by_group[group]
            print(f"  {group}: {m.num_scenes} scenes")
    print()

    # Generate LaTeX table
    latex_output = generate_latex_table(metrics_by_group, args.stage, args.output)
    
    # Print to stdout
    print("=" * 80)
    print("LaTeX Table:")
    print("=" * 80)
    print(latex_output)
    print("=" * 80)


if __name__ == "__main__":
    main()
