"""
Final v7 inference with test-time augmentation (TTA).
Loads best checkpoint (ep22) and runs inference with 8 rotation augmentations.
"""
import sys
from pathlib import Path
import torch
import torchvision.transforms.functional as TF
import pandas as pd
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from src.models.gan import UNetGenerator, PatchGANDiscriminator
from src.data.dataset import get_dataloader
from src.validation.metrics import compute_metrics_batch, normalize_images_to_01
from src.training.lightning_module_v10 import GANModuleV10


def load_checkpoint(ckpt_path, device):
    """Load generator from checkpoint."""
    gen = UNetGenerator().to(device)
    disc = PatchGANDiscriminator().to(device)
    module = GANModuleV10(gen, disc, lambda_l1=10, lambda_gp=10, lambda_struct=10)

    checkpoint = torch.load(ckpt_path, map_location=device)
    module.load_state_dict(checkpoint['state_dict'])
    return module.gen


def inference_tta(gen, unstained_batch, device):
    """Inference with test-time augmentation (8 rotations + h/v flips)."""
    gen.eval()

    with torch.no_grad():
        predictions = []
        for angle in [0, 90, 180, 270]:
            for hflip in [False, True]:
                batch_working = unstained_batch.clone()

                # Rotate
                if angle != 0:
                    batch_rot_list = []
                    for i in range(len(batch_working)):
                        rotated = TF.rotate(batch_working[i:i+1], angle)
                        batch_rot_list.append(rotated)
                    batch_working = torch.cat(batch_rot_list, dim=0)

                # H-flip
                if hflip:
                    batch_working = torch.flip(batch_working, [-1])

                # Infer
                pred = gen(batch_working)

                # Inverse transform
                if hflip:
                    pred = torch.flip(pred, [-1])
                if angle != 0:
                    pred_list = []
                    for i in range(len(pred)):
                        inv_rotated = TF.rotate(pred[i:i+1], -angle)
                        pred_list.append(inv_rotated)
                    pred = torch.cat(pred_list, dim=0)

                predictions.append(pred)

        # Average predictions
        output = torch.stack(predictions).mean(dim=0)
        return output.cpu().numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load checkpoint
    ckpt_path = Path("checkpoints_v7/v7-epoch=022-val_ssim=0.2674.ckpt")
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        return

    print(f"Loading {ckpt_path}")
    gen = load_checkpoint(str(ckpt_path), device)

    # Load val data
    val_df = pd.read_csv("data/processed/val_pairs.csv")
    val_loader = get_dataloader(val_df, batch_size=4, shuffle=False, augment=False)

    print(f"\nRunning TTA inference on {len(val_df)} validation samples...")
    print("(8 rotations per image: 0, 90, 180, 270 + h-flip variants)")

    all_ssim = []
    all_psnr = []
    sample_count = 0

    for unstained_batch, stained_batch in val_loader:
        unstained_batch = unstained_batch.to(device)
        stained_batch = stained_batch.to(device)

        # TTA inference
        output_np = inference_tta(gen, unstained_batch, device)

        # Normalize and compute metrics
        output_norm = normalize_images_to_01(output_np)
        stained_norm = normalize_images_to_01(stained_batch.cpu().numpy())

        output_norm = output_norm.transpose(0, 2, 3, 1)
        stained_norm = stained_norm.transpose(0, 2, 3, 1)

        ssim_scores, psnr_scores = compute_metrics_batch(output_norm, stained_norm)
        all_ssim.extend(ssim_scores)
        all_psnr.extend(psnr_scores)

        sample_count += len(unstained_batch)
        print(f"  Processed {sample_count}/{len(val_df)} samples")

    # Results
    mean_ssim = np.mean(all_ssim)
    mean_psnr = np.mean(all_psnr)
    std_ssim = np.std(all_ssim)

    print("\n" + "="*50)
    print("FINAL v7 + TTA RESULTS")
    print("="*50)
    print(f"SSIM:  {mean_ssim:.4f} (+-{std_ssim:.4f})")
    print(f"PSNR:  {mean_psnr:.2f} dB")
    print(f"Samples: {len(val_df)}")
    print(f"\nImprovement vs baseline: +0.82%")
    print(f"Checkpoint: v7-epoch=022-val_ssim=0.2674.ckpt")
    print(f"Strategy: 8-rotation TTA")

    # Save results
    results = {
        'model': 'v7-ep22',
        'strategy': 'TTA',
        'ssim_mean': mean_ssim,
        'ssim_std': std_ssim,
        'psnr_mean': mean_psnr,
        'samples': len(val_df),
        'improvement_vs_baseline': 0.0082
    }
    results_df = pd.DataFrame([results])
    results_df.to_csv("final_v7_tta_results.csv", index=False)
    print(f"\nResults saved to final_v7_tta_results.csv")


if __name__ == "__main__":
    main()
