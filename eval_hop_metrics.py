"""Evaluate recursive IDJSCC hops on the fixed, balanced CIFAR10 subset."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from cifar10_msssim import MS_SSIM
from torch.utils.data import DataLoader, TensorDataset

from app_common import build_model, seed_initial
from sim_engine import _split_model_output, _stabilize_recursive_output


ROOT = Path(__file__).resolve().parent
METRICS = ("psnr_db", "ms_ssim", "lpips")


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/cifar10_test_100/subset.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cr", type=float, default=3)
    parser.add_argument("--snr", type=float, default=18)
    parser.add_argument("--M", type=int, default=0)
    parser.add_argument("--x_min", type=float, default=-3.0)
    parser.add_argument("--x_max", type=float, default=3.0)
    parser.add_argument("--channel", choices=("awgn", "rayleigh"), default="awgn")
    parser.add_argument("--hops", type=int, default=30)
    parser.add_argument("--test_batch_size", type=int, default=10)
    parser.add_argument("--group_num", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13890)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser.parse_args()


def load_subset(path):
    with np.load(path, allow_pickle=False) as subset:
        images = subset["images"].copy()
        labels = subset["labels"].copy()
        indices = subset["indices"].copy()
    if images.shape != (100, 32, 32, 3) or images.dtype != np.uint8:
        raise ValueError("Expected 100 uint8 RGB images of size 32x32")
    if labels.shape != (100,) or indices.shape != (100,):
        raise ValueError("Expected 100 labels and test indices")
    if not np.array_equal(np.unique(labels, return_counts=True),
                          np.stack((np.arange(10), np.full(10, 10)))):
        raise ValueError("Expected exactly 10 images per class")
    if len(np.unique(indices)) != 100 or np.any((indices < 0) | (indices >= 10000)):
        raise ValueError("Expected unique indices from the CIFAR10 test split")
    return TensorDataset(
        torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255,
        torch.from_numpy(labels).long(), torch.from_numpy(indices).long(),
    )


def image_metrics(reconstruction, reference, perceptual, ms_ssim):
    # Match utils.calc_psnr: round to uint8 values; perfect matches score 100 dB.
    error = (torch.round(reconstruction * 255) - torch.round(reference * 255)).square()
    mse = error.flatten(1).mean(1)
    psnr = torch.where(mse == 0, 100.0, 10 * torch.log10(255 ** 2 / mse))
    # Swin MS_SSIM returns 1 - similarity, with one value per image.
    structural = 1 - ms_ssim(reference, reconstruction)
    # Same metric and clamping as Swin main_lpips.calc_test_alex_lpips.
    # Keep per-image scores; their mean matches the reference's default reduction.
    perceptual.reset()
    distance = perceptual(reconstruction.clamp(0., 1.), reference.clamp(0., 1.)).flatten()
    perceptual.reset()
    values = torch.stack((psnr, structural, distance), dim=1)
    if not torch.isfinite(values).all():
        raise RuntimeError("Non-finite image quality metric")
    return values


def write_csv(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_metric_vectors(output_dir, summary):
    """Export hop-ordered vectors; round only the copyable text output."""
    ordered = sorted(summary, key=lambda row: int(row["hop"]))
    vectors = {metric: [float(row[metric]) for row in ordered] for metric in METRICS}
    lines = []
    for metric, heading, digits in zip(METRICS, ("PSNR", "MS-SSIM", "LPIPS"), (2, 3, 6)):
        values = ", ".join(f"{value:.{digits}f}" for value in vectors[metric])
        lines.append(f"{heading} = [{values}]")
    text = "\n".join(lines) + "\n"
    (output_dir / "metric_vectors.txt").write_text(text)
    (output_dir / "metric_vectors.json").write_text(json.dumps(vectors, indent=2) + "\n")
    return text


def main(args):
    if min(args.hops, args.test_batch_size, args.group_num, args.cpu_threads) < 1:
        raise ValueError("Hops, batch size, group count and CPU threads must be positive")
    if args.M < 0 or args.x_min >= args.x_max:
        raise ValueError("Require M >= 0, x_min < x_max")
    if not args.resume.is_file():
        raise FileNotFoundError(args.resume)
    dataset = load_subset(args.data)
    # Never overwrite an earlier experiment's results.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.set_num_threads(args.cpu_threads)
    seed_initial(args.seed)
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        ms_ssim = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).to(device).eval()
    except ImportError as exc:
        raise RuntimeError("Install torchmetrics with image dependencies in the selected Python environment") from exc
    model = build_model("Inverse", args.cr, args.channel, device, use_denoise=True,
                        x_min=args.x_min, x_max=args.x_max).to(device).eval()
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    state = checkpoint.get("net", checkpoint)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()}, strict=True)
    perceptual = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True, reduction="none").to(device).eval()
    perceptual.requires_grad_(False)
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(device=device, samples=len(dataset),
                  data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                  checkpoint_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest(),
                  reference="original image at every hop",
                  hop_input="round(clamp(reconstruction, 0, 1) * 255) / 255",
                  psnr="RGB 32x32, uint8-rounded pixels; exact match = 100 dB",
                  ms_ssim="native RGB 32x32 [0,1], Swin MS_SSIM, window_size=3, levels=4, data_range=1, normalized weights; similarity=1-loss",
                  lpips="torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity, net_type=alex, normalize=True, reduction=none; native RGB 32x32, clamp [0,1], per-image arithmetic mean; matches Swin main_lpips.py testing metric",
                  metric_reconstruction="clamped float output before hop-input rounding",
                  torch_version=torch.__version__)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    rows = []
    nonfinite_batches = 0
    with torch.inference_mode():
        for group in range(args.group_num):
            # Model/metric construction must not consume the channel's random sequence.
            seed_initial(args.seed + group)
            loader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=0)
            for batch, (reference, labels, indices) in enumerate(loader, start=1):
                reference = reference.to(device)
                current = reference
                for hop in range(1, args.hops + 1):
                    reconstruction, _ = _split_model_output(model(current, snr=args.snr, M=args.M))
                    reconstruction, finite = _stabilize_recursive_output(reconstruction)
                    nonfinite_batches += int(not finite)
                    values = image_metrics(reconstruction, reference, perceptual, ms_ssim).cpu().numpy()
                    for label, index, scores in zip(labels.tolist(), indices.tolist(), values):
                        rows.append(dict(group=group + 1, hop=hop, test_index=index, label=label,
                                         **dict(zip(METRICS, map(float, scores)))))
                    current = torch.round(reconstruction * 255) / 255
                print(f"Group {group + 1}/{args.group_num}, batch {batch}/{len(loader)} complete", flush=True)
    write_csv(args.output_dir / "per_image.csv", ("group", "hop", "test_index", "label", *METRICS), rows)
    summary = []
    for hop in range(1, args.hops + 1):
        selected = [row for row in rows if row["hop"] == hop]
        average = {metric: float(np.mean([row[metric] for row in selected])) for metric in METRICS}
        summary.append(dict(hop=hop, samples=len(selected), **average))
        print(f"Hop {hop:02d}: PSNR={average['psnr_db']:.6f} dB  "
              f"MS-SSIM={average['ms_ssim']:.6f}  LPIPS={average['lpips']:.6f}")
    write_csv(args.output_dir / "per_hop.csv", ("hop", "samples", *METRICS), summary)
    config["nonfinite_output_batches"] = nonfinite_batches
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Results: {args.output_dir.resolve()}")
    print(write_metric_vectors(args.output_dir, summary), end="", flush=True)


if __name__ == "__main__":
    main(get_args())
