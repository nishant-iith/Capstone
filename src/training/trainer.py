import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.training.lightning_module import Pix2PixLightning
from src.data.dataset import get_dataloader
import pandas as pd

def train_model(train_csv, val_csv, epochs=100):
    # Load pairing data
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    train_loader = get_dataloader(train_df, batch_size=4)
    val_loader = get_dataloader(val_df, batch_size=4, shuffle=False)
    
    model = Pix2PixLightning()
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="best-pix2pix",
        save_top_k=1,
        mode="min"
    )
    
    early_stop = EarlyStopping(monitor="val_loss", patience=10)
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stop]
    )
    
    trainer.fit(model, train_loader, val_loader)
    return model

if __name__ == "__main__":
    # This would be run on Kaggle
    print("Trainer initialized. Ready for Kaggle execution.")
