import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.models.gan import UNetGenerator, PatchGANDiscriminator
from src.training.structural_loss import StructuralConsistencyLoss

class Pix2PixLightning(pl.LightningModule):
    """
    Pix2Pix Training Orchestrator using PyTorch Lightning.
    Implements LSGAN (MSE) loss, L1 structural loss, and Structural Consistency loss.
    """
    def __init__(self, lr=2e-4, lambda_l1=100, lambda_struct=10):
        super().__init__()
        self.save_hyperparameters()

        self.gen = UNetGenerator()
        self.disc = PatchGANDiscriminator()

        self.criterion_gan = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.criterion_struct = StructuralConsistencyLoss()

    def forward(self, x):
        return self.gen(x)

    def training_step(self, batch, batch_idx):
        unstained, stained = batch

        # --- Generate fake stained image ---
        fake_stained = self.gen(unstained)

        # --- Train Discriminator ---
        # PatchGAN takes (condition, target) — unstained is the condition
        real_pred = self.disc(unstained, stained)
        fake_pred = self.disc(unstained, fake_stained.detach())

        real_loss = self.criterion_gan(real_pred, torch.ones_like(real_pred))
        fake_loss = self.criterion_gan(fake_pred, torch.zeros_like(fake_pred))
        d_loss = (real_loss + fake_loss) / 2

        # --- Train Generator ---
        # Adversarial: fool the discriminator into predicting 1 for fake
        fake_pred_for_g = self.disc(unstained, fake_stained)
        g_gan_loss = self.criterion_gan(fake_pred_for_g, torch.ones_like(fake_pred_for_g))
        # L1 loss: pixel-level similarity to ground truth
        g_l1_loss = self.criterion_l1(fake_stained, stained)
        # Structural consistency: edge alignment to suppress hallucinations
        g_struct_loss = self.criterion_struct(fake_stained, stained)

        # Three-term loss (Phases 2 + 3)
        g_loss = g_gan_loss + self.hparams.lambda_l1 * g_l1_loss + self.hparams.lambda_struct * g_struct_loss

        # Log all components
        self.log("d_loss",    d_loss,        prog_bar=True)
        self.log("g_loss",    g_loss,        prog_bar=True)
        self.log("g_gan",     g_gan_loss,    prog_bar=False)
        self.log("g_l1",      g_l1_loss,     prog_bar=False)
        self.log("g_struct",  g_struct_loss, prog_bar=True)

        return {"loss": g_loss}

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.gen.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.disc.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        
        return [opt_g, opt_d], []
