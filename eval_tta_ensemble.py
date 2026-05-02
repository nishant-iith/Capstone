"""
Evaluate v20_fixed and v21A-Hibou-B with base inference, 4-flip TTA, and ensemble.

Uses the exact clean v20/v21 validation split:
- top 1000 rows from registered_pairs_all.csv
- seed 42 permutation
- last 100 indices as validation
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
OUT_CSV = "logs/tta_ensemble_eval.csv"
OUT_SUMMARY = "logs/tta_ensemble_eval_summary.txt"


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
    state = torch.load(V20_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_v21():
    model = HibouBInputSkipUNet("histai/hibou-b").to(DEVICE)
    state = torch.load(V21_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
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


def invert_aug(x, aug):
    return apply_aug(x, aug)


@torch.no_grad()
def predict_base(model, x):
    return model(x).clamp(0, 1)


@torch.no_grad()
def predict_tta4(model, x):
    preds = []
    for aug in ("id", "h", "v", "hv"):
        pred = model(apply_aug(x, aug)).clamp(0, 1)
        preds.append(invert_aug(pred, aug))
    return torch.stack(preds, dim=0).mean(dim=0).clamp(0, 1)


def compute_metrics(pred, target):
    pred_np = pred.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    target_np = target.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    ssim = float(ssim_fn(target_np, pred_np, channel_axis=2, data_range=1.0))
    psnr = float(psnr_fn(target_np, pred_np, data_range=1.0))
    pcc = float(np.corrcoef(pred_np.ravel(), target_np.ravel())[0, 1])
    return ssim, psnr, pcc


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

    methods = [
        "v20_base",
        "v20_tta4",
        "v21_base",
        "v21_tta4",
        "ens_base",
        "ens_tta4",
    ]
    accum = {name: {"ssim": [], "psnr": [], "pcc": []} for name in methods}

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prefix", "method", "ssim", "psnr", "pcc"])
        for idx, (prefixes, unstained, stained) in enumerate(loader, start=1):
            unstained = unstained.to(DEVICE, non_blocking=True)
            stained = stained.to(DEVICE, non_blocking=True)
            prefix = prefixes[0]

            v20_base = predict_base(v20, unstained)
            v20_tta4 = predict_tta4(v20, unstained)
            v21_base = predict_base(v21, unstained)
            v21_tta4 = predict_tta4(v21, unstained)
            ens_base = ((v20_base + v21_base) * 0.5).clamp(0, 1)
            ens_tta4 = ((v20_tta4 + v21_tta4) * 0.5).clamp(0, 1)

            preds = {
                "v20_base": v20_base,
                "v20_tta4": v20_tta4,
                "v21_base": v21_base,
                "v21_tta4": v21_tta4,
                "ens_base": ens_base,
                "ens_tta4": ens_tta4,
            }
            for name, pred in preds.items():
                ssim, psnr, pcc = compute_metrics(pred, stained)
                accum[name]["ssim"].append(ssim)
                accum[name]["psnr"].append(psnr)
                accum[name]["pcc"].append(pcc)
                writer.writerow([prefix, name, f"{ssim:.6f}", f"{psnr:.6f}", f"{pcc:.6f}"])

            if idx % 10 == 0:
                best_so_far = max(
                    (np.mean(accum[name]["ssim"]), name)
                    for name in methods
                    if accum[name]["ssim"]
                )
                print(f"{idx:3d}/{len(ds)} done | best so far {best_so_far[1]} {best_so_far[0]:.4f}")

    lines = []
    lines.append(f"Evaluated {len(ds)} validation pairs in {(time.time() - t0):.1f}s")
    lines.append("")
    lines.append(f"{'method':<12} {'SSIM':>8} {'PSNR':>8} {'PCC':>8}")
    lines.append("-" * 40)
    for name in methods:
        ssim = np.mean(accum[name]["ssim"])
        psnr = np.mean(accum[name]["psnr"])
        pcc = np.mean(accum[name]["pcc"])
        lines.append(f"{name:<12} {ssim:8.4f} {psnr:8.2f} {pcc:8.4f}")

    summary = "\n".join(lines)
    print(summary)
    with open(OUT_SUMMARY, "w") as f:
        f.write(summary + "\n")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
