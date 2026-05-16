"""
Compute PSNR, SSIM, and LPIPS on the test set of a trained PartNet-Video scene.

Reads the best iteration from ``<model>/results.json`` (``axis_eval`` block from
``eval_axis.py``), then the latest PLY checkpoint.

Merges metrics into ``<model>/results.json`` under ``image_metrics`` (does not
wipe ``axis_eval`` or other keys). Also prints the summary to stdout.

Example::

    python eval_image_metrics.py -m output/PartNetVideo/Toilet_103227
"""

from __future__ import annotations

import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path

import torch

from utils.general_utils import safe_state, build_rotation
from utils.image_utils import psnr as psnr_fn
from utils.loss_utils import ssim as ssim_fn
from scene import Scene
from gaussian_renderer import GaussianModel, render
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.results_json import read_best_iteration, save_results_json_merged


def _resolve_iteration(model_path: Path, override: int | None) -> int:
    """Pick iteration: explicit flag > results.json > latest PLY."""
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
        raise RuntimeError(f"No point_cloud/iteration_* directories under {pc_root}")
    return iters[-1]


def _make_lpips():
    """Return an ``lpips(a, b) -> float`` callable, or None if lpips is unavailable."""
    try:
        import lpips as lpips_module
    except Exception as exc:
        print(f"[WARN] lpips not available ({exc}); LPIPS will be reported as NaN.",
              file=sys.stderr)
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = lpips_module.LPIPS(net="alex").cuda().eval()

    @torch.no_grad()
    def _fn(a: torch.Tensor, b: torch.Tensor) -> float:
        # a, b: [3, H, W] in [0, 1] → [-1, 1], [1, 3, H, W]
        a = (a.unsqueeze(0) * 2.0 - 1.0).clamp(-1.0, 1.0)
        b = (b.unsqueeze(0) * 2.0 - 1.0).clamp(-1.0, 1.0)
        return float(net(a, b).item())

    return _fn


def _mean(xs):
    xs = [x for x in xs if x is not None and x == x]  # drop None / NaN
    return float(sum(xs) / len(xs)) if xs else float("nan")


def evaluate(args) -> int:
    model_path = Path(args.model_path)
    dataset = ModelParams_for_args.extract(args)
    pipeline = PipelineParams_for_args.extract(args)

    iteration = _resolve_iteration(
        model_path,
        args.iteration if args.iteration is not None and args.iteration > 0 else None,
    )
    print(f"Evaluating image metrics at iteration {iteration} for {model_path}")

    ckpt_path = model_path / "ckpts" / f"ours_{iteration}.pth"
    if not ckpt_path.is_file():
        print(f"[ERROR] missing checkpoint {ckpt_path}", file=sys.stderr)
        return 1

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        ckpt = torch.load(str(ckpt_path), map_location="cuda")
        art_params = ckpt["articulation_params"]
        art_R = art_params["art_R"].cuda()
        art_T = art_params["art_T"].cuda()
        articulation_matrix = build_rotation(art_R)

        # Hard-assign each Gaussian to its argmax part (mirrors render_video.py).
        articulation_weights = gaussians.get_weight
        max_indices = torch.argmax(articulation_weights, dim=1)
        hardened = torch.zeros_like(articulation_weights)
        hardened[torch.arange(articulation_weights.shape[0]), max_indices] = 1.0

        bg = [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0]
        background = torch.tensor(bg, dtype=torch.float32, device="cuda")

        lpips_fn = _make_lpips()

        test_views = scene.getTestCameras()
        if not test_views:
            print("[INFO] No test cameras (train.py was likely run without --eval); skipping.")
            return 0

        # Per-view records: (image_name, time, psnr, ssim, lpips)
        records: list[tuple[str, float, float, float, float]] = []
        for view in test_views:
            pkg = render(
                view, gaussians, pipeline, background,
                articulation_weights=hardened,
                articulation_matrix=articulation_matrix,
                articulation_trans=art_T,
            )
            rendered = torch.clamp(pkg["render"], 0.0, 1.0)
            gt = torch.clamp(view.original_image[:3].to("cuda"), 0.0, 1.0)

            p = float(psnr_fn(rendered.unsqueeze(0), gt.unsqueeze(0)).mean().item())
            s = float(ssim_fn(rendered.unsqueeze(0), gt.unsqueeze(0)).item())
            l = lpips_fn(rendered, gt) if lpips_fn is not None else float("nan")

            t = float(getattr(view, "time", 0.0))
            records.append((view.image_name, t, p, s, l))

    psnr_all  = [r[2] for r in records]
    ssim_all  = [r[3] for r in records]
    lpips_all = [r[4] for r in records]
    start = [r for r in records if r[1] == 0.0]
    end   = [r for r in records if r[1] == 1.0]

    psnr_mean,  ssim_mean,  lpips_mean  = _mean(psnr_all),  _mean(ssim_all),  _mean(lpips_all)
    psnr_start = _mean([r[2] for r in start])
    ssim_start = _mean([r[3] for r in start])
    lpips_start = _mean([r[4] for r in start])
    psnr_end   = _mean([r[2] for r in end])
    ssim_end   = _mean([r[3] for r in end])
    lpips_end  = _mean([r[4] for r in end])

    lines = [
        "",
        f"--- Image metrics (test set, iter {iteration}, "
        f"N={len(records)}: start={len(start)}, end={len(end)}) ---",
        f"PSNR mean:  {psnr_mean:.4f}",
        f"SSIM mean:  {ssim_mean:.4f}",
        f"LPIPS mean: {lpips_mean:.4f}",
    ]
    if start:
        lines.append(
            "start (time=0.0): "
            f"PSNR={psnr_start:.4f}  SSIM={ssim_start:.4f}  LPIPS={lpips_start:.4f}"
        )
    if end:
        lines.append(
            "end   (time=1.0): "
            f"PSNR={psnr_end:.4f}  SSIM={ssim_end:.4f}  LPIPS={lpips_end:.4f}"
        )
    print("\n".join(lines) + "\n")

    image_metrics: dict = {
        "iteration": int(iteration),
        "num_views": len(records),
        "num_start_views": len(start),
        "num_end_views": len(end),
        "psnr_mean": psnr_mean,
        "ssim_mean": ssim_mean,
        "lpips_mean": lpips_mean,
        "lpips_network": "alex",
    }
    if start:
        image_metrics["start"] = {
            "psnr": psnr_start,
            "ssim": ssim_start,
            "lpips": lpips_start,
        }
    if end:
        image_metrics["end"] = {
            "psnr": psnr_end,
            "ssim": ssim_end,
            "lpips": lpips_end,
        }

    save_results_json_merged(model_path, {"image_metrics": image_metrics})

    return 0


# argparse plumbing -----------------------------------------------------------
# ``ModelParams``/``PipelineParams`` register their own fields on the parser and
# expose an ``.extract(args)`` method. We use module-level references so the
# ``evaluate`` function can stay readable.
ModelParams_for_args: ModelParams
PipelineParams_for_args: PipelineParams


def main() -> int:
    global ModelParams_for_args, PipelineParams_for_args
    parser = ArgumentParser(description="Compute PSNR/SSIM/LPIPS on the test set.")
    ModelParams_for_args  = ModelParams(parser, sentinel=True)
    PipelineParams_for_args = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int,
                        help="Force iteration; default reads results.json / latest PLY.")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    safe_state(args.quiet)
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
