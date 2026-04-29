"""TTA evaluation for v19b checkpoint."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from skimage import io
from skimage.metrics import structural_similarity as ssim_fn
import segmentation_models_pytorch as smp
import warnings
warnings.filterwarnings('ignore')

REGISTERED_CSV = "data/processed/registered_pairs_all.csv"
MODEL_PATH = "models/v19b_model.pth"
SEED = 42
TOP_N = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ValDataset(Dataset):
    def __init__(self, csv_file, top_n, seed=42):
        df = pd.read_csv(csv_file).head(top_n).reset_index(drop=True)
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(df))
        n_train = int(0.9 * len(df))
        val_indices = indices[n_train:]
        self.images = []
        for idx in val_indices:
            row = df.iloc[idx]
            try:
                s = io.imread(row["stained"]).astype(np.float32) / 255.0
                u = io.imread(row["unstained"]).astype(np.float32) / 255.0
                if len(s.shape) == 2: s = np.stack([s]*3, axis=-1)
                if len(u.shape) == 2: u = np.stack([u]*3, axis=-1)
                self.images.append((s, u))
            except Exception:
                continue

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        s, u = self.images[idx]
        s_t = torch.from_numpy(s).permute(2, 0, 1)
        u_t = torch.from_numpy(u).permute(2, 0, 1)
        return u_t, s_t


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                              in_channels=3, classes=3, activation=None)

    def forward(self, x):
        return torch.sigmoid(self.model(x))


def predict_tta(model, x):
    """4-flip TTA: orig, hflip, vflip, hflip+vflip — average in output space."""
    with torch.no_grad():
        p0 = model(x)
        p1 = torch.flip(model(torch.flip(x, [3])), [3])   # hflip
        p2 = torch.flip(model(torch.flip(x, [2])), [2])   # vflip
        p3 = torch.flip(model(torch.flip(x, [2, 3])), [2, 3])  # both
    return (p0 + p1 + p2 + p3) / 4.0


def evaluate(model, loader, use_tta):
    model.eval()
    ssims = []
    for u, s in loader:
        u, s = u.to(DEVICE), s.to(DEVICE)
        with torch.no_grad():
            pred = predict_tta(model, u) if use_tta else model(u)
        for b in range(pred.shape[0]):
            p_np = pred[b].cpu().numpy().transpose(1, 2, 0)
            s_np = s[b].cpu().numpy().transpose(1, 2, 0)
            ssims.append(ssim_fn(p_np, s_np, channel_axis=2, data_range=1.0))
    return np.mean(ssims), np.std(ssims)


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print("Loading model...")
    model = Model().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print("Model loaded.")

    print("Loading val data...")
    val_ds = ValDataset(REGISTERED_CSV, TOP_N, seed=SEED)
    loader = DataLoader(val_ds, batch_size=4, num_workers=0, pin_memory=True)
    print(f"Val size: {len(val_ds)}")

    print("\nBaseline (no TTA)...")
    base_mean, base_std = evaluate(model, loader, use_tta=False)
    print(f"  SSIM: {base_mean:.4f} ± {base_std:.4f}")

    print("\nTTA (4-flip)...")
    tta_mean, tta_std = evaluate(model, loader, use_tta=True)
    print(f"  SSIM: {tta_mean:.4f} ± {tta_std:.4f}")

    delta = tta_mean - base_mean
    print(f"\n{'='*40}")
    print(f"Baseline : {base_mean:.4f}")
    print(f"TTA      : {tta_mean:.4f}")
    print(f"Delta    : {delta:+.4f} ({delta/base_mean*100:+.2f}%)")
    print(f"{'='*40}")
