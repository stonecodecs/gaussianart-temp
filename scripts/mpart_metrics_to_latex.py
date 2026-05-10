#!/usr/bin/env python3
"""
Build a LaTeX table from mpart90_metrics_summary.json (from aggregate_mpart_results.py).

Columns: 2 parts | 3 parts | 4–5 parts | 6–20 parts | All (global_scene_level)
Rows: axis angle (°), axis position (line distance, same scale as eval_axis), part motion (°)

Merged buckets use weighted-by-num_scenes means; std is a mixture-style pooled std from
per-bucket scene-level mean ± std (see _merge_scene_level).

LaTeX: add \\usepackage{booktabs} for \\toprule / \\midrule / \\bottomrule.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def _merge_scene_level(
    buckets: list[tuple[int, dict[str, Any]]],
) -> dict[str, float]:
    """
    buckets: list of (num_scenes, scene_level dict with *_avg and *_std)
    Returns merged angle_mean_avg/std, distance_mean_avg/std, theta_diff_mean_avg/std.
    """
    total_n = sum(n for n, _ in buckets if n > 0)
    if total_n == 0:
        return {
            "angle_mean_avg": float("nan"),
            "angle_mean_std": float("nan"),
            "distance_mean_avg": float("nan"),
            "distance_mean_std": float("nan"),
            "theta_diff_mean_avg": float("nan"),
            "theta_diff_mean_std": float("nan"),
        }

    def merge_pair(avg_key: str, std_key: str) -> tuple[float, float]:
        mu_w = 0.0
        second_moment = 0.0
        for n, sl in buckets:
            if n <= 0:
                continue
            mu = float(sl[avg_key])
            sig = float(sl[std_key])
            v = sig * sig
            mu_w += n * mu
            second_moment += n * (v + mu * mu)
        mu_comb = mu_w / total_n
        var_comb = second_moment / total_n - mu_comb * mu_comb
        std_comb = math.sqrt(max(0.0, var_comb))
        return mu_comb, std_comb

    aa, asd = merge_pair("angle_mean_avg", "angle_mean_std")
    da, dsd = merge_pair("distance_mean_avg", "distance_mean_std")
    ta, tsd = merge_pair("theta_diff_mean_avg", "theta_diff_mean_std")

    return {
        "angle_mean_avg": aa,
        "angle_mean_std": asd,
        "distance_mean_avg": da,
        "distance_mean_std": dsd,
        "theta_diff_mean_avg": ta,
        "theta_diff_mean_std": tsd,
    }


def _fmt_pm(mean: float, std: float, decimals: int = 2) -> str:
    if math.isnan(mean):
        return "---"
    if math.isnan(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def build_table(data: dict[str, Any], decimals: int = 2) -> str:
    by_parts: dict[str, dict[str, Any]] = data["by_parts_num"]
    global_sl = data["global_scene_level"]

    def bucket(key: str) -> tuple[int, dict[str, Any]] | None:
        if key not in by_parts:
            return None
        b = by_parts[key]
        return b["num_scenes"], b["scene_level"]

    def merge_keys(keys: list[str]) -> dict[str, float]:
        buckets: list[tuple[int, dict[str, Any]]] = []
        for k in keys:
            t = bucket(k)
            if t is not None and t[0] > 0:
                buckets.append(t)
        return _merge_scene_level(buckets)

    col_2 = merge_keys(["2"])
    col_3 = merge_keys(["3"])
    col_45 = merge_keys(["4", "5"])

    keys_6_20 = [str(k) for k in range(6, 21) if str(k) in by_parts]
    col_620 = merge_keys(keys_6_20)

    col_all = {
        "angle_mean_avg": global_sl["angle_mean_avg"],
        "angle_mean_std": global_sl["angle_mean_std"],
        "distance_mean_avg": global_sl["distance_mean_avg"],
        "distance_mean_std": global_sl["distance_mean_std"],
        "theta_diff_mean_avg": global_sl["theta_diff_mean_avg"],
        "theta_diff_mean_std": global_sl["theta_diff_mean_std"],
    }

    cols = [col_2, col_3, col_45, col_620, col_all]
    headers = [
        "2 parts",
        "3 parts",
        "4--5 parts",
        "6--20 parts",
        "All",
    ]

    lines: list[str] = []
    lines.append(r"\begin{tabular}{l" + "c" * len(headers) + "}")
    lines.append(r"\toprule")
    lines.append(" & " + " & ".join(headers) + r" \\")
    lines.append(r"\midrule")

    row_specs = [
        (r"Axis angle ($^\circ$)", "angle_mean_avg", "angle_mean_std"),
        ("Axis pos.", "distance_mean_avg", "distance_mean_std"),
        (r"Part motion ($^\circ$)", "theta_diff_mean_avg", "theta_diff_mean_std"),
    ]

    for label, ak, sk in row_specs:
        cells = [_fmt_pm(c[ak], c[sk], decimals) for c in cols]
        lines.append(label + " & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="MPArt metrics JSON → LaTeX table")
    p.add_argument(
        "--json",
        type=Path,
        default=Path("output/mpart90_metrics_summary.json"),
        help="Path to mpart90_metrics_summary.json",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Write LaTeX here (default: print to stdout)",
    )
    p.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Decimal places for numbers",
    )
    p.add_argument(
        "--wrap",
        action="store_true",
        help="Wrap in minimal table environment (requires booktabs)",
    )
    args = p.parse_args()

    path = args.json.resolve()
    if not path.is_file():
        print(f"Not found: {path}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(path.read_text(encoding="utf-8"))
    body = build_table(data, decimals=args.decimals)

    if args.wrap:
        out = "\n".join(
            [
                r"\begin{table}[t]",
                r"\centering",
                r"\small",
                body,
                r"\caption{GaussianArt axis metrics by part count (scene-level means; mean $\pm$ std across scenes).}",
                r"\label{tab:mpart-axis-metrics}",
                r"\end{table}",
            ]
        )
    else:
        out = body

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(out + "\n", encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(out)


if __name__ == "__main__":
    main()
