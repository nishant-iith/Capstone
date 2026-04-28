"""
CycleGAN training for unpaired histology virtual staining.
"""
import sys
from pathlib import Path
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.insert(0, str(Path(__file__).parent))

from src.models.cyclegan import UNetGenerator, CycleGANDiscriminator
from src.data.dataset import get_cyclegan_dataloader
from src.training.lightning_module_cyclegan import CycleGANModule

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # We can use the same registered CSV for paths, but data loader treats them as unpaired
    df = pd.read_csv("data/processed/registered_pairs.csv")
    
    # Split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]

    train_loader = get_cyclegan_dataloader(train_df, batch_size=1, shuffle=True, augment=True)
    # Validation for CycleGAN is mostly visual, but we can log cycle consistency
    val_loader = get_cyclegan_dataloader(val_df, batch_size=1, shuffle=False, augment=False)

    # Models
    g_ab = UNetGenerator(pretrained=True)
    g_ba = UNetGenerator(pretrained=True)
    d_a = CycleGANDiscriminator()
    d_b = CycleGANDiscriminator()

    module = CycleGANModule(g_ab, g_ba, d_a, d_b)

    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints_cyclegan",
        filename="cycle-{epoch:03d}",
        save_top_k=2,
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_cb],
    )

    trainer.fit(module, train_loader, val_loader)

if __name__ == "__main__":
    main()
