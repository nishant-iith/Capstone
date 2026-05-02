
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from skimage import io
import segmentation_models_pytorch as smp
import warnings
from src.validation.metrics import compute_metrics_batch

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

def evaluate(model, loader):
    model.eval()
    all_ssim, all_psnr, all_pcc = [], [], []

    for u, s in loader:
        u, s = u.to(DEVICE), s.to(DEVICE)
        with torch.no_grad():
            pred = model(u)

        # Convert to numpy [B, H, W, 3] in [0, 1]
        p_np = pred.cpu().numpy().transpose(0, 2, 3, 1)
        s_np = s.cpu().numpy().transpose(0, 2, 3, 1)

        ssims, psnrs, pccs = compute_metrics_batch(p_np, s_np)
        all_ssim.extend(ssims)
        all_psnr.extend(psnrs)
        all_pcc.extend(pccs)

    return (np.mean(all_ssim), np.std(all_ssim),
            np.mean(all_psnr), np.std(all_psnr),
            np.mean(all_pcc), np.std(all_pcc))

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

    print("\nEvaluating v19b...")
    m_ssim, s_ssim, m_psnr, s_psnr, m_pcc, s_pcc = evaluate(model, loader)

    print(f"\n{'='*40}")
    print(f"v19b Evaluation Results:")
    print(f"SSIM: {m_ssim:.4f} ± {s_ssim:.4f}")
    print(f"PSNR: {m_psnr:.2f} ± {s_psnr:.2f} dB")
    print(f"PCC:  {m_pcc:.4f} ± {s_pcc:.4f}")
    print(f"{'='*40}")
