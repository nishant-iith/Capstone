import sys
import os

# Add root to path
sys.path.append(os.getcwd())

try:
    from src.data.pairing import get_paired_patches
    from src.data.splitting import split_by_patient
    from src.data.augmentation import TissueAugmenter
    from PIL import Image
    print("Imports successful!")
    
    # 1. Test Pairing
    s_dir = "1000-20260412T223922Z-3-001/1000/tes_stain"
    u_dir = "1000-20260412T223922Z-3-001/1000/tes_unstain"
    df = get_paired_patches(s_dir, u_dir)
    print(f"✓ Pairing: Found {len(df)} pairs.")
    
    # 2. Test Splitting
    train, val, test = split_by_patient(df)
    print(f"✓ Splitting: Train({len(train)}), Val({len(val)}), Test({len(test)})")
    
    # 3. Test Augmentation
    sample = df.iloc[0]
    s_img = Image.open(sample['stained']).convert('RGB')
    u_img = Image.open(sample['unstained']).convert('RGB')
    
    augmenter = TissueAugmenter()
    s_aug, u_aug = augmenter(s_img, u_img)
    print(f"✓ Augmentation: Produced tensors of size {s_aug.shape}")
    
    print("\nPhase 1 smoke test PASSED.")
except Exception as e:
    print(f"Smoke test FAILED: {e}")
