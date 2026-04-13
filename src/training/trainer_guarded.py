"""
Fine-tuning trainer for guarded Pix2Pix model.

This script loads a Phase 2 baseline checkpoint and fine-tunes it on validation data
using the three-term loss (LSGAN + L1 + Structural Consistency) to create a guarded
model that suppresses hallucinations.

Usage:
    python -c "from src.training.trainer_guarded import train_guarded_model; \
      train_guarded_model(baseline_ckpt_path='checkpoints/best-pix2pix.ckpt', \
        train_csv='data/processed/train_pairs.csv', \
        val_csv='data/processed/val_pairs.csv', epochs=50)"
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.training.lightning_module import Pix2PixLightning
from src.data.dataset import get_dataloader
import pandas as pd


def train_guarded_model(baseline_ckpt_path, train_csv, val_csv, epochs=50):
    """
    Fine-tune Phase 2 baseline on validation data using structural loss to suppress hallucinations.

    This function:
    1. Loads Phase 2 baseline checkpoint via load_from_checkpoint
    2. Transfers weights to Phase 3 model with lambda_struct=10
    3. Fine-tunes on validation split with three-term loss (LSGAN + L1 + Structural)
    4. Monitors all three loss terms on console
    5. Saves best checkpoint based on total loss

    Args:
        baseline_ckpt_path (str): Path to Phase 2 baseline checkpoint
        train_csv (str): Path to training CSV with image pairs
        val_csv (str): Path to validation CSV with image pairs
        epochs (int): Number of epochs to fine-tune (default: 50)

    Returns:
        Pix2PixLightning: Trained model instance

    Raises:
        FileNotFoundError: If checkpoint or CSV files don't exist
    """
    # Load pairing data
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # Create dataloaders
    train_loader = get_dataloader(train_df, batch_size=4)
    val_loader = get_dataloader(val_df, batch_size=4, shuffle=False)

    # Load Phase 2 baseline and add structural loss weight
    model = Pix2PixLightning.load_from_checkpoint(
        baseline_ckpt_path,
        lr=2e-4,
        lambda_l1=100,
        lambda_struct=10  # NEW: structural loss weight for hallucination suppression
    )

    # Setup callbacks for checkpointing and early stopping
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="phase3_guarded_model",
        save_top_k=1,
        mode="min"
    )

    # Allow longer convergence with new loss term
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=15,
        mode="min"
    )

    # Create trainer with console logging
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stop],
        precision="16-mixed",  # Mixed precision for Kaggle T4
        log_every_n_steps=10,  # Frequent logging to observe loss terms
    )

    # Run fine-tuning
    trainer.fit(model, train_loader, val_loader)

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tune Phase 2 baseline on validation data with structural loss"
    )
    parser.add_argument(
        "--baseline",
        default="checkpoints/best-pix2pix.ckpt",
        help="Path to Phase 2 baseline checkpoint"
    )
    parser.add_argument(
        "--train-csv",
        required=True,
        help="Path to training CSV with image pairs"
    )
    parser.add_argument(
        "--val-csv",
        required=True,
        help="Path to validation CSV with image pairs"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs to fine-tune"
    )

    args = parser.parse_args()

    model = train_guarded_model(
        args.baseline,
        args.train_csv,
        args.val_csv,
        epochs=args.epochs
    )

    print(f"Guarded model saved to checkpoints/phase3_guarded_model.ckpt")
