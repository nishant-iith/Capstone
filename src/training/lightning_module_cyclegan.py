"""
CycleGAN Lightning Module for unpaired H&E virtual staining.
G_AB: Unstained (A) -> Stained (B)
G_BA: Stained (B) -> Unstained (A)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import itertools

class CycleGANModule(pl.LightningModule):
    def __init__(self, g_ab, g_ba, d_a, d_b, lambda_cycle=10, lambda_id=5):
        super().__init__()
        self.g_ab = g_ab
        self.g_ba = g_ba
        self.d_a = d_a
        self.d_b = d_b
        self.lambda_cycle = lambda_cycle
        self.lambda_id = lambda_id
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        # Unpaired data: (A, B) might not be the same patch
        real_a, real_b = batch
        opt_g, opt_d = self.optimizers()

        # ─── G Step ───
        # Identity Loss: G_AB(real_b) -> real_b; G_BA(real_a) -> real_a
        id_a = self.g_ba(real_a)
        id_b = self.g_ab(real_b)
        loss_id_a = F.l1_loss(id_a, real_a) * self.lambda_id
        loss_id_b = F.l1_loss(id_b, real_b) * self.lambda_id

        # GAN Loss
        fake_b = self.g_ab(real_a)
        pred_fake_b = self.d_b(fake_b)
        loss_gan_ab = F.mse_loss(pred_fake_b, torch.ones_like(pred_fake_b))

        fake_a = self.g_ba(real_b)
        pred_fake_a = self.d_a(fake_a)
        loss_gan_ba = F.mse_loss(pred_fake_a, torch.ones_like(pred_fake_a))

        # Cycle Loss
        cycle_a = self.g_ba(fake_b)
        loss_cycle_a = F.l1_loss(cycle_a, real_a) * self.lambda_cycle

        cycle_b = self.g_ab(fake_a)
        loss_cycle_b = F.l1_loss(cycle_b, real_b) * self.lambda_cycle

        g_loss = loss_gan_ab + loss_gan_ba + loss_cycle_a + loss_cycle_b + loss_id_a + loss_id_b
        
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        # ─── D Step ───
        # D_A
        pred_real_a = self.d_a(real_a)
        loss_d_a_real = F.mse_loss(pred_real_a, torch.ones_like(pred_real_a))
        pred_fake_a = self.d_a(fake_a.detach())
        loss_d_a_fake = F.mse_loss(pred_fake_a, torch.zeros_like(pred_fake_a))
        loss_d_a = (loss_d_a_real + loss_d_a_fake) * 0.5

        # D_B
        pred_real_b = self.d_b(real_b)
        loss_d_b_real = F.mse_loss(pred_real_b, torch.ones_like(pred_real_b))
        pred_fake_b = self.d_b(fake_b.detach())
        loss_d_b_fake = F.mse_loss(pred_fake_b, torch.zeros_like(pred_fake_b))
        loss_d_b = (loss_d_b_real + loss_d_b_fake) * 0.5

        opt_d.zero_grad()
        self.manual_backward(loss_d_a + loss_d_b)
        opt_d.step()

        # Logging
        self.log_dict({
            "g_loss": g_loss,
            "d_loss": loss_d_a + loss_d_b,
            "cycle_a": loss_cycle_a,
            "cycle_b": loss_cycle_b
        }, prog_bar=True)

    def configure_optimizers(self):
        params_g = itertools.chain(self.g_ab.parameters(), self.g_ba.parameters())
        params_d = itertools.chain(self.d_a.parameters(), self.d_b.parameters())
        opt_g = torch.optim.Adam(params_g, lr=2e-4, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(params_d, lr=2e-4, betas=(0.5, 0.999))
        return [opt_g, opt_d]
