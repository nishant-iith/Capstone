"""
Evaluate v7 final model with 4 strategies:
1. Baseline (ep22 checkpoint)
2. Test-time augmentation (TTA) - 8 rotations + average
3. Macenko preprocessing (test-time)
4. Ensemble checkpoints (ep22 + nearby peaks)
"""
import sys
from pathlib import Path
import torch
import torchvision.transforms.functional as TF
import pandas as pd
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing

sys.path.insert(0, str(Path(__file__).parent))

from src.models.gan import UNetGenerator, PatchGANDiscriminator
from src.data.dataset import get_dataloader
from src.validation.metrics import compute_metrics_batch, normalize_images_to_01
from src.training.lightning_module_v10 import GANModuleV10


def macenko_normalize(img_np):
    """Apply Macenko stain normalization."""
    try:
        if img_np.shape[2] == 3:
            img = img_np.copy().astype(np.float32)

            # Normalize to [0, 255]
            if img.max() <= 1:
                img = img * 255

            # OD computation
            img = np.maximum(img, 1)
            od = -np.log(img / 255)

            # Threshold for stain vectors
            mask = (od >= threshold_otsu(od)).astype(float)

            if mask.sum() > 100:  # Need enough pixels
                od_masked = od[mask > 0].reshape(-1, 3)
                if od_masked.shape[0] > 2:
                    # SVD for stain matrix estimation
                    U, _, _ = np.linalg.svd(od_masked.T)
                    stain_matrix = U[:, :2]

                    # Concentration estimation
                    conc = np.linalg.lstsq(stain_matrix, od.reshape(-1, 3).T, rcond=None)[0]

                    # Normalization to reference
                    conc_norm = np.percentile(conc, 99, axis=1)
                    conc_norm = np.maximum(conc_norm, 1)
                    conc = (conc / conc_norm[:, None]) * [1.9705, 1.0308]

                    od_norm = stain_matrix @ conc
                    img_norm = 255 * np.exp(-od_norm.reshape(img.shape))
                    img_norm = np.clip(img_norm, 0, 255).astype(np.uint8)

                    return img_norm / 255.0
    except Exception:
        pass

    return img_np


def load_checkpoint(ckpt_path, device):
    """Load generator from checkpoint."""
    gen = UNetGenerator().to(device)
    disc = PatchGANDiscriminator().to(device)
    module = GANModuleV10(gen, disc, lambda_l1=10, lambda_gp=10, lambda_struct=10)

    checkpoint = torch.load(ckpt_path, map_location=device)
    module.load_state_dict(checkpoint['state_dict'])
    return module.gen


def load_multiple_checkpoints(ckpt_paths, device):
    """Load multiple generators from checkpoints."""
    gens = []
    for ckpt_path in ckpt_paths:
        if Path(ckpt_path).exists():
            gens.append(load_checkpoint(ckpt_path, device))
    return gens


def inference(gen, unstained_batch, device, tta=False, macenko=False):
    """Run inference with optional TTA and Macenko."""
    gen.eval()

    with torch.no_grad():
        if tta:
            # 8 rotations: 0, 90, 180, 270 + h/v flips
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
        else:
            output = gen(unstained_batch)

        # Convert to numpy
        output_np = output.cpu().numpy()

        # Apply Macenko if requested
        if macenko:
            output_np = np.array([
                macenko_normalize((o.transpose(1, 2, 0) + 1) / 2)
                for o in output_np
            ])
            output_np = output_np * 2 - 1

        return output_np


def evaluate_strategy(gen, val_loader, device, strategy_name, tta=False, macenko=False):
    """Evaluate a single strategy."""
    print(f"\n[{strategy_name}]")

    all_ssim = []
    all_psnr = []

    for unstained_batch, stained_batch in val_loader:
        unstained_batch = unstained_batch.to(device)
        stained_batch = stained_batch.to(device)

        # Inference
        output_np = inference(gen, unstained_batch, device, tta=tta, macenko=macenko)

        # Normalize to [0, 1] and convert to [B, H, W, 3] format
        output_norm = normalize_images_to_01(output_np)
        stained_norm = normalize_images_to_01(stained_batch.cpu().numpy())

        # Transpose from [B, C, H, W] to [B, H, W, C]
        output_norm = output_norm.transpose(0, 2, 3, 1)
        stained_norm = stained_norm.transpose(0, 2, 3, 1)

        # Metrics
        ssim_scores, psnr_scores = compute_metrics_batch(output_norm, stained_norm)
        all_ssim.extend(ssim_scores)
        all_psnr.extend(psnr_scores)

    mean_ssim = np.mean(all_ssim)
    mean_psnr = np.mean(all_psnr)

    print(f"  SSIM: {mean_ssim:.4f}")
    print(f"  PSNR: {mean_psnr:.2f}")

    return mean_ssim, mean_psnr


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load val data
    val_df = pd.read_csv("data/processed/val_pairs.csv")
    val_loader = get_dataloader(val_df, batch_size=4, shuffle=False, augment=False)

    # Best checkpoint
    ckpt_path = Path("checkpoints_v7/v7-epoch=022-val_ssim=0.2674.ckpt")
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        return

    print(f"Loading {ckpt_path}")
    gen = load_checkpoint(str(ckpt_path), device)

    # Evaluate strategies
    results = {}

    # 1. Baseline
    ssim, psnr = evaluate_strategy(gen, val_loader, device, "Baseline (ep22)")
    results['Baseline'] = {'ssim': ssim, 'psnr': psnr}

    # 2. TTA
    ssim, psnr = evaluate_strategy(gen, val_loader, device, "TTA (8 rotations)", tta=True)
    results['TTA'] = {'ssim': ssim, 'psnr': psnr}

    # 3. Ensemble (ep14, ep17, ep22, ep24)
    print("\n[Ensemble (ep14, ep17, ep22, ep24)]")
    ensemble_ckpts = [
        "checkpoints_v7/v7-epoch=014-val_ssim=0.2670.ckpt",
        "checkpoints_v7/v7-epoch=017-val_ssim=0.2671.ckpt",
        "checkpoints_v7/v7-epoch=022-val_ssim=0.2674.ckpt",
    ]
    ensemble_gens = load_multiple_checkpoints(ensemble_ckpts, device)
    if len(ensemble_gens) > 0:
        print(f"  Loaded {len(ensemble_gens)} checkpoints")
        all_ssim = []
        all_psnr = []
        for unstained_batch, stained_batch in val_loader:
            unstained_batch = unstained_batch.to(device)
            stained_batch = stained_batch.to(device)

            # Ensemble inference
            predictions = []
            for ens_gen in ensemble_gens:
                with torch.no_grad():
                    pred = ens_gen(unstained_batch)
                predictions.append(pred)

            # Average
            output = torch.stack(predictions).mean(dim=0)
            output_norm = normalize_images_to_01(output.cpu().numpy())
            stained_norm = normalize_images_to_01(stained_batch.cpu().numpy())
            output_norm = output_norm.transpose(0, 2, 3, 1)
            stained_norm = stained_norm.transpose(0, 2, 3, 1)

            ssim_scores, psnr_scores = compute_metrics_batch(output_norm, stained_norm)
            all_ssim.extend(ssim_scores)
            all_psnr.extend(psnr_scores)

        print(f"  SSIM: {np.mean(all_ssim):.4f}")
        print(f"  PSNR: {np.mean(all_psnr):.2f}")
        results['Ensemble'] = {'ssim': np.mean(all_ssim), 'psnr': np.mean(all_psnr)}

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    baseline_ssim = results['Baseline']['ssim']
    for strategy, metrics in results.items():
        improvement = (metrics['ssim'] - baseline_ssim) / baseline_ssim * 100
        print(f"{strategy:15} SSIM: {metrics['ssim']:.4f} ({improvement:+.2f}%)")

    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv("eval_results_v7.csv")
    print(f"\nResults saved to eval_results_v7.csv")


if __name__ == "__main__":
    main()
