"""
Weighted v20/v21 TTA ensemble sweep on the clean validation split.

Sweeps v20 weight from 0.00 to 1.00 for:
- base predictions
- 4-flip TTA predictions

Writes a compact CSV of weight-level metrics.
"""

import csv
import os
import time
import warnings

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from torch.utils.data import DataLoader, Dataset

from train_v20 import ConvNeXtUNet
from train_v21a_hibou_b import HibouBInputSkipUNet

warnings.filterwarnings("ignore")


REGISTERED_CSV = "data/processed/registered_pairs_all.csv"
TOP_N = 1000
SEED = 42
BATCH_SIZE = 1
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

V20_PATH = "models/v20_fixed_model.pth"
V21_PATH = "models/v21a_hibou_b_model.pth"
OUT_CSV = "logs/weighted_ensemble_sweep.csv"
OUT_SUMMARY = "logs/weighted_ensemble_sweep_summary.txt"

WEIGHTS = [round(x, 2) for x in np.linspace(0.0, 1.0, 21)]


class ValPairsDataset(Dataset):
    def __init__(self, csv_file, top_n, indices):
        df = pd.read_csv(csv_file).head(top_n).reset_index(drop=True)
        self.rows = df.iloc[indices].reset_index(drop=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        stained = io.imread(row["stained"]).astype(np.float32) / 255.0
        unstained = io.imread(row["unstained"]).astype(np.float32) / 255.0
        if stained.ndim == 2:
            stained = np.stack([stained] * 3, axis=-1)
        if unstained.ndim == 2:
            unstained = np.stack([unstained] * 3, axis=-1)
        stained = torch.from_numpy(stained.copy()).permute(2, 0, 1).contiguous()
        unstained = torch.from_numpy(unstained.copy()).permute(2, 0, 1).contiguous()
        return row["prefix"], unstained, stained


def clean_val_indices():
    df = pd.read_csv(REGISTERED_CSV).head(TOP_N).reset_index(drop=True)
    split_perm = np.random.RandomState(SEED).permutation(len(df))
    return split_perm[int(0.9 * len(df)) :]


def load_v20():
    model = ConvNeXtUNet().to(DEVICE)
    model.load_state_dict(torch.load(V20_PATH, map_location=DEVICE), strict=True)
    model.eval()
    return model


def load_v21():
    model = HibouBInputSkipUNet("histai/hibou-b").to(DEVICE)
    model.load_state_dict(torch.load(V21_PATH, map_location=DEVICE), strict=True)
    model.eval()
    return model


def apply_aug(x, aug):
    if aug == "id":
        return x
    if aug == "h":
        return torch.flip(x, dims=[3])
    if aug == "v":
        return torch.flip(x, dims=[2])
    if aug == "hv":
        return torch.flip(x, dims=[2, 3])
    raise ValueError(aug)


@torch.no_grad()
def predict_base(model, x):
    return model(x).clamp(0, 1)


@torch.no_grad()
def predict_tta4(model, x):
    preds = []
    for aug in ("id", "h", "v", "hv"):
        pred = model(apply_aug(x, aug)).clamp(0, 1)
        preds.append(apply_aug(pred, aug))
    return torch.stack(preds, dim=0).mean(dim=0).clamp(0, 1)


def compute_metrics_np(pred_np, target_np):
    ssim = float(ssim_fn(target_np, pred_np, channel_axis=2, data_range=1.0))
    psnr = float(psnr_fn(target_np, pred_np, data_range=1.0))
    pcc = float(np.corrcoef(pred_np.ravel(), target_np.ravel())[0, 1])
    return ssim, psnr, pcc


def to_np(tensor):
    return tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)


def main():
    os.makedirs("logs", exist_ok=True)
    val_indices = clean_val_indices()
    ds = ValPairsDataset(REGISTERED_CSV, TOP_N, val_indices)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    t0 = time.time()
    print(f"Loading models on {DEVICE}...")
    v20 = load_v20()
    v21 = load_v21()

    accum = {
        mode: {w: {"ssim": [], "psnr": [], "pcc": []} for w in WEIGHTS}
        for mode in ("base", "tta4")
    }

    with torch.no_grad():
        for idx, (_, unstained, stained) in enumerate(loader, start=1):
            unstained = unstained.to(DEVICE, non_blocking=True)
            stained_np = to_np(stained)

            v20_base = to_np(predict_base(v20, unstained))
            v21_base = to_np(predict_base(v21, unstained))
            v20_tta4 = to_np(predict_tta4(v20, unstained))
            v21_tta4 = to_np(predict_tta4(v21, unstained))

            for weight in WEIGHTS:
                for mode, p20, p21 in (
                    ("base", v20_base, v21_base),
                    ("tta4", v20_tta4, v21_tta4),
                ):
                    pred = np.clip(weight * p20 + (1.0 - weight) * p21, 0.0, 1.0)
                    ssim, psnr, pcc = compute_metrics_np(pred, stained_np)
                    accum[mode][weight]["ssim"].append(ssim)
                    accum[mode][weight]["psnr"].append(psnr)
                    accum[mode][weight]["pcc"].append(pcc)

            if idx % 10 == 0:
                best = max(
                    (
                        np.mean(accum[mode][weight]["ssim"]),
                        mode,
                        weight,
                    )
                    for mode in accum
                    for weight in WEIGHTS
                    if accum[mode][weight]["ssim"]
                )
                print(f"{idx:3d}/{len(ds)} done | best {best[1]} w20={best[2]:.2f} SSIM={best[0]:.4f}")

    rows = []
    for mode in ("base", "tta4"):
        for weight in WEIGHTS:
            metrics = accum[mode][weight]
            rows.append(
                {
                    "mode": mode,
                    "v20_weight": weight,
                    "v21_weight": round(1.0 - weight, 2),
                    "ssim": float(np.mean(metrics["ssim"])),
                    "psnr": float(np.mean(metrics["psnr"])),
                    "pcc": float(np.mean(metrics["pcc"])),
                }
            )

    rows.sort(key=lambda row: (row["ssim"], row["psnr"], row["pcc"]), reverse=True)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["mode", "v20_weight", "v21_weight", "ssim", "psnr", "pcc"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "mode": row["mode"],
                    "v20_weight": f"{row['v20_weight']:.2f}",
                    "v21_weight": f"{row['v21_weight']:.2f}",
                    "ssim": f"{row['ssim']:.6f}",
                    "psnr": f"{row['psnr']:.6f}",
                    "pcc": f"{row['pcc']:.6f}",
                }
            )

    lines = []
    lines.append(f"Evaluated {len(ds)} validation pairs in {(time.time() - t0):.1f}s")
    lines.append("")
    lines.append("Top weighted ensembles")
    lines.append(f"{'rank':>4} {'mode':<6} {'w20':>5} {'w21':>5} {'SSIM':>8} {'PSNR':>8} {'PCC':>8}")
    lines.append("-" * 54)
    for rank, row in enumerate(rows[:10], start=1):
        lines.append(
            f"{rank:4d} {row['mode']:<6} {row['v20_weight']:5.2f} {row['v21_weight']:5.2f} "
            f"{row['ssim']:8.4f} {row['psnr']:8.2f} {row['pcc']:8.4f}"
        )

    summary = "\n".join(lines)
    print(summary)
    with open(OUT_SUMMARY, "w") as f:
        f.write(summary + "\n")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
