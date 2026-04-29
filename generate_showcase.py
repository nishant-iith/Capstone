"""
Generate showcase images: Unstained | Virtually Stained | Real Stained
Uses v19b model (best SSIM 0.7489) on top 20 SSIM pairs.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/v19b_model.pth"
REGISTERED_CSV = "data/processed/registered_pairs_all.csv"
OUTPUT_DIR = "showcase_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class DenseUNetGenerator(nn.Module):
    def __init__(self):
        import segmentation_models_pytorch as smp
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=3,
            activation=None
        )
    
    def forward(self, x):
        return self.model(x)

def load_model():
    model = DenseUNetGenerator().to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def get_top_n_pairs(csv_path, n=20):
    df = pd.read_csv(csv_path)
    df = df.sort_values('ssim', ascending=False)
    return df.head(n)

def create_composite(pairs, model, output_path):
    n = len(pairs)
    rows = 4
    cols = 5
    
    fig, axes = plt.subplots(rows, cols * 3, figsize=(25, 20))
    fig.suptitle("Unstained (Input) | Virtually Stained (v19b Output) | Real Stained (Ground Truth)", fontsize=16, fontweight='bold')
    
    for idx, (_, row) in enumerate(pairs.iterrows()):
        row_idx = idx // cols
        col_idx = idx % cols
        
        unstained_path = row['unstained']
        stained_path = row['stained']
        
        unstained = np.array(Image.open(unstained_path).convert('RGB'))
        stained = np.array(Image.open(stained_path).convert('RGB'))
        
        with torch.no_grad():
            input_tensor = torch.from_numpy(unstained).permute(2, 0, 1).float() / 255.0
            input_tensor = input_tensor.unsqueeze(0).to(DEVICE)
            output = model(input_tensor)
            output = torch.sigmoid(output[0]).permute(1, 2, 0).cpu().numpy()
            virtual_stained = (output * 255).astype(np.uint8)
        
        ssim_val = row['ssim']
        img_name = row['prefix']
        
        for col_offset, (img, title) in enumerate([
            (unstained, "Unstained"),
            (virtual_stained, f"Virtual (v19b)"),
            (stained, "Real Stained")
        ]):
            ax = axes[row_idx, col_idx * 3 + col_offset]
            ax.imshow(img)
            if row_idx == 0:
                ax.set_title(title, fontsize=11, fontweight='bold')
            ax.axis('off')
        
        axes[row_idx, col_idx * 3 + 1].set_xlabel(f"SSIM: {ssim_val:.4f}", fontsize=9)
        
        if col_idx == 0:
            axes[row_idx, 0].set_ylabel(img_name[:25], fontsize=8, rotation=0, labelpad=40)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    print("Loading v19b model (best SSIM: 0.7489)...")
    model = load_model()
    
    print("Getting top 20 SSIM pairs...")
    pairs = get_top_n_pairs(REGISTERED_CSV, n=20)
    print(f"Selected {len(pairs)} pairs")
    
    print("\nGenerating composite image...")
    output_path = os.path.join(OUTPUT_DIR, "top20_ssim_showcase.png")
    create_composite(pairs, model, output_path)
    
    print("\nDone!")

if __name__ == "__main__":
    main()