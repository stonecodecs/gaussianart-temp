"""
Compute PSNR, SSIM, and LPIPS on the test set of a trained PartNet-Video scene.

Reads the best iteration from ``<model>/results.txt`` (written by ``eval_axis.py``).
Falls back to the latest checkpoint when ``results.txt`` is missing. Loads the
trained Gaussians + articulation parameters, renders each test view (start-state
views render canonically; end-state views render with articulation applied
automatically inside ``gaussian_renderer.render``), then averages the metrics.

Appends a results block to ``<model>/results.txt`` and prints the same numbers
to stdout. Designed to be called by ``train_eval_all_pnv.py`` as a separate
pipeline stage.

Example::

    python eval_image_metrics.py -m output/PartNetVideo/Toilet_103227
"""

from __future__ import annotations

import os
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


def _resolve_iteration(model_path: Path, override: int | None) -> int:
    """Pick the iteration to evaluate: explicit > results.txt 'The best:' > latest PLY."""
    if override is not None and override > 0:
        return override

    results = model_path / "results.txt"
    if results.is_file():
        try:
            with results.open() as f:
                first = f.readline()
            return int(first.split(":")[-1].strip())
        except Exception:
            pass

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

    block = []
    block.append("")
    block.append(
        f"--- Image metrics (test set, iter {iteration}, "
        f"N={len(records)}: start={len(start)}, end={len(end)}) ---"
    )
    block.append(f"PSNR mean:  {psnr_mean:.4f}")
    block.append(f"SSIM mean:  {ssim_mean:.4f}")
    block.append(f"LPIPS mean: {lpips_mean:.4f}")
    if start:
        block.append(
            f"start (time=0.0): PSNR={psnr_start:.4f}  SSIM={ssim_start:.4f}  LPIPS={lpips_start:.4f}"
        )
    if end:
        block.append(
            f"end   (time=1.0): PSNR={psnr_end:.4f}  SSIM={ssim_end:.4f}  LPIPS={lpips_end:.4f}"
        )

    text = "\n".join(block) + "\n"
    print(text)
    results_path = model_path / "results.txt"
    _append_metrics_block(results_path, text)

    return 0


_METRICS_HEADER = "--- Image metrics (test set,"


def _append_metrics_block(results_path: Path, block: str) -> None:
    """Append ``block`` to ``results.txt``, replacing any prior image-metrics block.

    Re-runs of this script should not accumulate duplicate metrics sections; we
    detect previous sections by the ``_METRICS_HEADER`` marker and drop them
    (plus the blank separator line directly above) before appending the new
    block.
    """
    if results_path.is_file():
        with results_path.open("r") as f:
            lines = f.readlines()
        kept: list[str] = []
        skipping = False
        for line in lines:
            if line.startswith(_METRICS_HEADER):
                skipping = True
                # Drop the trailing blank separator that precedes the section.
                while kept and kept[-1].strip() == "":
                    kept.pop()
                continue
            if skipping:
                # Stop skipping at the next blank line that follows the section,
                # since each section is followed by a blank line.
                if line.strip() == "":
                    skipping = False
                continue
            kept.append(line)
        with results_path.open("w") as f:
            f.writelines(kept)
    with results_path.open("a") as f:
        f.write(block)


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
                        help="Force a specific iteration; default reads 'The best' from results.txt.")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    safe_state(args.quiet)
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
