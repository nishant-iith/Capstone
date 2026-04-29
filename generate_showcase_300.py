"""
Generate individual showcase images for top 300 SSIM pairs.
Each image shows: Unstained | Virtually Stained | Real Stained triplet.
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
OUTPUT_DIR = "showcase_images_300"
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

def get_top_n_pairs(csv_path, n=300):
    df = pd.read_csv(csv_path)
    df = df.sort_values('ssim', ascending=False)
    return df.head(n)

def create_triplet_image(unstained, virtual_stained, real_stained, img_name, ssim_val, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    titles = ["Unstained (Input)", "Virtually Stained (v19b)", "Real Stained (GT)"]
    images = [unstained, virtual_stained, real_stained]
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
    
    fig.suptitle(f"{img_name} | SSIM: {ssim_val:.4f}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

def main():
    print("Loading v19b model...")
    model = load_model()
    
    print("Getting top 300 SSIM pairs...")
    pairs = get_top_n_pairs(REGISTERED_CSV, n=300)
    print(f"Selected {len(pairs)} pairs")
    
    print("\nGenerating individual triplet images...")
    for idx, (_, row) in enumerate(pairs.iterrows()):
        unstained_path = row['unstained']
        stained_path = row['stained']
        img_name = row['prefix']
        ssim_val = row['ssim']
        
        unstained = np.array(Image.open(unstained_path).convert('RGB'))
        stained = np.array(Image.open(stained_path).convert('RGB'))
        
        with torch.no_grad():
            input_tensor = torch.from_numpy(unstained).permute(2, 0, 1).float() / 255.0
            input_tensor = input_tensor.unsqueeze(0).to(DEVICE)
            output = model(input_tensor)
            output = torch.sigmoid(output[0]).permute(1, 2, 0).cpu().numpy()
            virtual_stained = (output * 255).astype(np.uint8)
        
        output_filename = f"{img_name}_ssim{ssim_val:.4f}.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        create_triplet_image(unstained, virtual_stained, stained, img_name, ssim_val, output_path)
        
        if (idx + 1) % 50 == 0:
            print(f"  Generated {idx + 1}/300 images...")
    
    print(f"\nDone! All 300 images saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()