"""
Final fixed evaluation with v20, v22A, and v21B.

Runs on the same 100 validation prefixes used by v20_fixed:
  - old_registered: original registered_pairs_all.csv rows
  - clahe_same_prefixes: CLAHE rows for those exact prefixes

Reports base, 4-flip TTA, and small weighted ensemble sweeps:
  - v20/v22
  - v22/v21B
  - v20/v22/v21B grid
"""

import csv
import os
import time
import warnings
from pathlib import Path

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


OLD_CSV = "data/processed/registered_pairs_all.csv"
CLAHE_CSV = "data/processed/registered_clahe_pairs_all.csv"
TOP_N = 1000
SEED = 42
BATCH_SIZE = 1
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

V20_PATH = "models/v20_fixed_model.pth"
V22_PATH = "models/v22a_content_quality_top1000_ft_model.pth"
V21B_PATH = "models/v21b_hibou_content_quality_ft_model.pth"

OUT_CSV = "logs/final_v21b_v22a_eval_per_pair.csv"
OUT_SUMMARY = "logs/final_v21b_v22a_eval_summary.txt"

PAIR_WEIGHTS = [0.00, 0.25, 0.50, 0.75, 1.00]
TRIPLE_WEIGHTS = [
    (0.20, 0.60, 0.20),
    (0.20, 0.70, 0.10),
    (0.25, 0.65, 0.10),
    (0.30, 0.60, 0.10),
    (0.30, 0.50, 0.20),
    (0.40, 0.50, 0.10),
]


class FixedRowsDataset(Dataset):
    def __init__(self, rows, dataset_name):
        self.rows = rows.reset_index(drop=True)
        self.dataset_name = dataset_name

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
        return self.dataset_name, row["prefix"], unstained, stained


def fixed_rows():
    old_df = pd.read_csv(OLD_CSV).head(TOP_N).reset_index(drop=True)
    split_perm = np.random.RandomState(SEED).permutation(len(old_df))
    val_indices = split_perm[int(0.9 * len(old_df)) :]
    old_rows = old_df.iloc[val_indices].reset_index(drop=True)
    clahe_df = pd.read_csv(CLAHE_CSV).set_index("prefix")
    missing = [prefix for prefix in old_rows["prefix"] if prefix not in clahe_df.index]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} CLAHE rows")
    clahe_rows = clahe_df.loc[old_rows["prefix"]].reset_index()
    return {"old_registered": old_rows, "clahe_same_prefixes": clahe_rows}


def load_v20_style(path):
    model = ConvNeXtUNet().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE), strict=True)
    model.eval()
    return model


def load_v21b(path):
    model = HibouBInputSkipUNet("histai/hibou-b").to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE), strict=True)
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


def to_np(tensor):
    return tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)


def metrics(pred_np, target_np):
    ssim = float(ssim_fn(target_np, pred_np, channel_axis=2, data_range=1.0))
    psnr = float(psnr_fn(target_np, pred_np, data_range=1.0))
    pcc = float(np.corrcoef(pred_np.ravel(), target_np.ravel())[0, 1])
    return ssim, psnr, pcc


def add(accum, dataset, method, pred_np, target_np):
    ssim, psnr, pcc = metrics(pred_np, target_np)
    key = (dataset, method)
    accum.setdefault(key, {"ssim": [], "psnr": [], "pcc": []})
    accum[key]["ssim"].append(ssim)
    accum[key]["psnr"].append(psnr)
    accum[key]["pcc"].append(pcc)
    return ssim, psnr, pcc


def main():
    for path in (V20_PATH, V22_PATH, V21B_PATH):
        if not Path(path).exists():
            raise FileNotFoundError(path)

    os.makedirs("logs", exist_ok=True)
    rows_by_dataset = fixed_rows()

    print(f"Loading models on {DEVICE}...")
    v20 = load_v20_style(V20_PATH)
    v22 = load_v20_style(V22_PATH)
    v21b = load_v21b(V21B_PATH)

    accum = {}
    t0 = time.time()

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "prefix", "method", "ssim", "psnr", "pcc"])

        for dataset_name, rows in rows_by_dataset.items():
            ds = FixedRowsDataset(rows, dataset_name)
            loader = DataLoader(
                ds,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                persistent_workers=NUM_WORKERS > 0,
            )
            print(f"\nEvaluating {dataset_name}: {len(ds)} pairs")

            for idx, (_, prefixes, unstained, stained) in enumerate(loader, start=1):
                unstained = unstained.to(DEVICE, non_blocking=True)
                target_np = to_np(stained)
                prefix = prefixes[0]

                p20_base = to_np(predict_base(v20, unstained))
                p22_base = to_np(predict_base(v22, unstained))
                p21b_base = to_np(predict_base(v21b, unstained))
                p20 = to_np(predict_tta4(v20, unstained))
                p22 = to_np(predict_tta4(v22, unstained))
                p21b = to_np(predict_tta4(v21b, unstained))

                direct = {
                    "v20_base": p20_base,
                    "v20_tta4": p20,
                    "v22_base": p22_base,
                    "v22_tta4": p22,
                    "v21b_base": p21b_base,
                    "v21b_tta4": p21b,
                }
                for method, pred_np in direct.items():
                    ssim, psnr, pcc = add(accum, dataset_name, method, pred_np, target_np)
                    writer.writerow([dataset_name, prefix, method, f"{ssim:.6f}", f"{psnr:.6f}", f"{pcc:.6f}"])

                for w22 in PAIR_WEIGHTS:
                    pred = np.clip((1.0 - w22) * p20 + w22 * p22, 0.0, 1.0)
                    method = f"ens20_22_w22_{w22:.2f}"
                    ssim, psnr, pcc = add(accum, dataset_name, method, pred, target_np)
                    writer.writerow([dataset_name, prefix, method, f"{ssim:.6f}", f"{psnr:.6f}", f"{pcc:.6f}"])

                for w21b in PAIR_WEIGHTS:
                    pred = np.clip((1.0 - w21b) * p22 + w21b * p21b, 0.0, 1.0)
                    method = f"ens22_21b_w21b_{w21b:.2f}"
                    ssim, psnr, pcc = add(accum, dataset_name, method, pred, target_np)
                    writer.writerow([dataset_name, prefix, method, f"{ssim:.6f}", f"{psnr:.6f}", f"{pcc:.6f}"])

                for w20, w22, w21b in TRIPLE_WEIGHTS:
                    pred = np.clip(w20 * p20 + w22 * p22 + w21b * p21b, 0.0, 1.0)
                    method = f"ens3_{w20:.2f}_{w22:.2f}_{w21b:.2f}"
                    ssim, psnr, pcc = add(accum, dataset_name, method, pred, target_np)
                    writer.writerow([dataset_name, prefix, method, f"{ssim:.6f}", f"{psnr:.6f}", f"{pcc:.6f}"])

                if idx % 10 == 0:
                    scoped = [
                        (float(np.mean(vals["ssim"])), method)
                        for (dataset, method), vals in accum.items()
                        if dataset == dataset_name
                    ]
                    best_ssim, best_method = max(scoped)
                    print(f"{idx:3d}/{len(ds)} done | best {best_method} SSIM={best_ssim:.4f}", flush=True)

    lines = [f"Evaluated fixed validation sets in {(time.time() - t0):.1f}s", ""]
    for dataset_name in rows_by_dataset:
        rows = []
        for (dataset, method), vals in accum.items():
            if dataset != dataset_name:
                continue
            rows.append(
                {
                    "method": method,
                    "ssim": float(np.mean(vals["ssim"])),
                    "psnr": float(np.mean(vals["psnr"])),
                    "pcc": float(np.mean(vals["pcc"])),
                }
            )
        rows.sort(key=lambda row: (row["ssim"], row["psnr"], row["pcc"]), reverse=True)
        lines.append(dataset_name)
        lines.append(f"{'rank':>4} {'method':<22} {'SSIM':>8} {'PSNR':>8} {'PCC':>8}")
        lines.append("-" * 58)
        for rank, row in enumerate(rows, start=1):
            lines.append(
                f"{rank:4d} {row['method']:<22} {row['ssim']:8.4f} {row['psnr']:8.2f} {row['pcc']:8.4f}"
            )
        lines.append("")

    summary = "\n".join(lines).rstrip()
    print("\n" + summary)
    with open(OUT_SUMMARY, "w") as f:
        f.write(summary + "\n")
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
