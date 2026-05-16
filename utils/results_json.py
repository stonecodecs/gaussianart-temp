"""
Merged ``results.json`` I/O shared by ``eval_axis.py``, ``eval_image_metrics.py``,
and training wrappers.

Each writer updates its own top-level key (``axis_eval``, ``image_metrics``,
``point_cloud_stats``, …) without discarding the rest of the file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RESULTS_JSON_NAME = "results.json"


def results_json_path(model_path: Path | str) -> Path:
    return Path(model_path).expanduser().resolve() / RESULTS_JSON_NAME


def load_results_json(model_path: Path | str) -> dict[str, Any]:
    p = results_json_path(model_path)
    if not p.is_file():
        return {}
    try:
        with p.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_results_json_merged(model_path: Path | str, updates: dict[str, Any]) -> None:
    """
    Merge ``updates`` into ``<model_path>/results.json``.

    - Top-level keys in ``updates`` overwrite existing keys of the same name.
    - For ``point_cloud_stats``, if both old and new values are dicts, merge
      dicts (so Gaussian count and PLY path can be updated separately).
    """
    mp = Path(model_path).expanduser().resolve()
    p = results_json_path(mp)
    merged = load_results_json(mp)

    for key, val in updates.items():
        if (
            key == "point_cloud_stats"
            and isinstance(val, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = {**merged[key], **val}
        else:
            merged[key] = val

    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, allow_nan=True)
        f.write("\n")
    tmp.replace(p)


def read_best_iteration(model_path: Path | str) -> int | None:
    """Best training checkpoint iteration from JSON (axis eval preferred)."""
    data = load_results_json(model_path)
    ae = data.get("axis_eval")
    if isinstance(ae, dict):
        for k in ("best_iteration", "best_iter"):
            if k in ae:
                try:
                    return int(ae[k])
                except (TypeError, ValueError):
                    pass
    im = data.get("image_metrics")
    if isinstance(im, dict) and "iteration" in im:
        try:
            return int(im["iteration"])
        except (TypeError, ValueError):
            pass
    return None
