"""
Generate visual comparison grids for current best validation setup.

Columns:
- Unstained input
- v20 base
- v20 4-flip TTA
- v21 4-flip TTA
- 55/45 weighted TTA ensemble
- Real H&E target

Rows are selected from the clean validation split using previous per-image
ensemble metrics: best, middle, and worst validation examples.
"""

import os
import warnings

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage import io
from skimage.metrics import structural_similarity as ssim_fn

from train_v20 import ConvNeXtUNet
from train_v21a_hibou_b import HibouBInputSkipUNet

warnings.filterwarnings("ignore")


REGISTERED_CSV = "data/processed/registered_pairs_all.csv"
METRICS_CSV = "logs/tta_ensemble_eval.csv"
TOP_N = 1000
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
V20_PATH = "models/v20_fixed_model.pth"
V21_PATH = "models/v21a_hibou_b_model.pth"
OUT_DIR = "showcase_images/v21_ensemble_comparison"
V20_WEIGHT = 0.55
V21_WEIGHT = 0.45


def clean_val_df():
    df = pd.read_csv(REGISTERED_CSV).head(TOP_N).reset_index(drop=True)
    split_perm = np.random.RandomState(SEED).permutation(len(df))
    val_indices = split_perm[int(0.9 * len(df)) :]
    return df.iloc[val_indices].reset_index(drop=True)


def load_rgb(path):
    img = io.imread(path).astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return np.clip(img, 0.0, 1.0)


def to_tensor(img):
    return torch.from_numpy(img.copy()).permute(2, 0, 1).unsqueeze(0).contiguous()


def to_img(tensor):
    img = tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(img, 0.0, 1.0)


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


def load_models():
    print(f"Loading models on {DEVICE}...")
    v20 = ConvNeXtUNet().to(DEVICE)
    v20.load_state_dict(torch.load(V20_PATH, map_location=DEVICE), strict=True)
    v20.eval()

    v21 = HibouBInputSkipUNet("histai/hibou-b").to(DEVICE)
    v21.load_state_dict(torch.load(V21_PATH, map_location=DEVICE), strict=True)
    v21.eval()
    return v20, v21


def selected_prefixes():
    metrics = pd.read_csv(METRICS_CSV)
    ens = metrics[metrics["method"] == "ens_tta4"].copy()
    ens = ens.sort_values("ssim", ascending=False).reset_index(drop=True)
    n = len(ens)
    groups = {
        "best": ens.head(8),
        "middle": ens.iloc[[max(0, n // 2 - 4 + i) for i in range(8)]],
        "worst": ens.tail(8).sort_values("ssim", ascending=True),
    }
    return {
        name: list(zip(group["prefix"].tolist(), group["ssim"].tolist()))
        for name, group in groups.items()
    }


def render_group(group_name, prefix_scores, val_df, v20, v21):
    rows = []
    for prefix, prior_ssim in prefix_scores:
        row = val_df[val_df["prefix"] == prefix]
        if row.empty:
            continue
        row = row.iloc[0]
        stained = load_rgb(row["stained"])
        unstained = load_rgb(row["unstained"])
        x = to_tensor(unstained).to(DEVICE)

        with torch.no_grad():
            v20_base = to_img(predict_base(v20, x))
            v20_tta = to_img(predict_tta4(v20, x))
            v21_tta = to_img(predict_tta4(v21, x))
            ensemble = np.clip(V20_WEIGHT * v20_tta + V21_WEIGHT * v21_tta, 0.0, 1.0)

        current_ssim = ssim_fn(stained, ensemble, channel_axis=2, data_range=1.0)
        rows.append(
            {
                "prefix": prefix,
                "prior_ssim": float(prior_ssim),
                "current_ssim": float(current_ssim),
                "images": [unstained, v20_base, v20_tta, v21_tta, ensemble, stained],
            }
        )

    if not rows:
        return None

    cols = ["Unstained", "v20", "v20 TTA", "v21 TTA", "55/45 Ens TTA", "Real H&E"]
    fig, axes = plt.subplots(
        len(rows),
        len(cols),
        figsize=(len(cols) * 2.4, len(rows) * 2.1),
        squeeze=False,
    )

    for r, item in enumerate(rows):
        for c, (title, img) in enumerate(zip(cols, item["images"])):
            ax = axes[r][c]
            ax.imshow(img)
            ax.axis("off")
            if r == 0:
                ax.set_title(title, fontsize=10)
        axes[r][0].set_ylabel(
            f"{item['prefix']}\nSSIM {item['current_ssim']:.4f}",
            fontsize=7,
            rotation=0,
            labelpad=72,
            va="center",
        )

    fig.suptitle(f"{group_name.title()} validation examples - weighted ensemble visual check", fontsize=14)
    fig.tight_layout(rect=[0.05, 0.0, 1.0, 0.98])
    out_path = os.path.join(OUT_DIR, f"{group_name}_grid.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_contact_sheet(paths):
    images = [Image.open(path).convert("RGB") for path in paths if path]
    if not images:
        return None
    width = max(img.width for img in images)
    total_height = sum(img.height for img in images)
    sheet = Image.new("RGB", (width, total_height), "white")
    y = 0
    for img in images:
        sheet.paste(img, (0, y))
        y += img.height
    out_path = os.path.join(OUT_DIR, "all_grids_contact_sheet.png")
    sheet.save(out_path)
    return out_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    val_df = clean_val_df()
    v20, v21 = load_models()
    groups = selected_prefixes()
    paths = []
    for group_name, prefix_scores in groups.items():
        print(f"Rendering {group_name} examples...")
        path = render_group(group_name, prefix_scores, val_df, v20, v21)
        if path:
            print(f"Saved {path}")
            paths.append(path)
    contact = save_contact_sheet(paths)
    if contact:
        print(f"Saved {contact}")


if __name__ == "__main__":
    main()
