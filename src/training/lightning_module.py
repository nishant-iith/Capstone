import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.models.gan import UNetGenerator, PatchGANDiscriminator

class Pix2PixLightning(pl.LightningModule):
    """
    Pix2Pix Training Orchestrator using PyTorch Lightning.
    Implements LSGAN (MSE) loss and L1 structural loss.
    """
    def __init__(self, lr=2e-4, lambda_l1=100):
        super().__init__()
        self.save_hyperparameters()
        
        self.gen = UNetGenerator()
        self.disc = PatchGANDiscriminator()
        
        self.criterion_gan = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()

    def forward(self, x):
        return self.gen(x)

    def training_step(self, batch, batch_idx):
        unstained, stained = batch
        
        # --- Train Discriminator ---
        self.disc.train()
        fake_stained = self.gen(unstained)
        
        # Real loss: how well it identifies real images as 1
        real_loss = self.criterion_gan(self.disc(stained), torch.ones_like(self.disc(stained)))
        # Fake loss: how well it identifies generated images as 0
        fake_loss = self.criterion_gan(self.disc(fake_stained.detach()), torch.zeros_like(self.disc(fake_stained)))
        
        d_loss = (real_loss + fake_loss) / 2
        
        # --- Train Generator ---
        self.gen.train()
        # Adversarial loss: trick discriminator into seeing 1
        g_gan_loss = self.criterion_gan(self.disc(fake_stained), torch.ones_like(self.disc(fake_stained)))
        # L1 loss: spatial similarity to target
        g_l1_loss = self.criterion_l1(fake_stained, stained)
        
        g_loss = g_gan_loss + self.hparams.lambda_l1 * g_l1_loss
        
        # Log and return
        self.log("d_loss", d_loss, prog_bar=True)
        self.log("g_loss", g_loss, prog_bar=True)
        self.log("g_l1", g_l1_loss, prog_bar=False)
        
        return {"loss": g_loss}

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.gen.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.disc.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        
        return [opt_g, opt_d], []
