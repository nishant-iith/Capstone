"""
Weakly Supervised Lightning Module.
Combines Unpaired CycleGAN losses with a small fraction of Paired L1 loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import itertools

class WeaklySupervisedModule(pl.LightningModule):
    def __init__(self, g_ab, g_ba, d_a, d_b, lambda_cycle=10, lambda_id=5, lambda_paired=50):
        super().__init__()
        self.save_hyperparameters(ignore=['g_ab', 'g_ba', 'd_a', 'd_b'])
        self.g_ab = g_ab
        self.g_ba = g_ba
        self.d_a = d_a
        self.d_b = d_b
        self.lambda_cycle = lambda_cycle
        self.lambda_id = lambda_id
        self.lambda_paired = lambda_paired
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        # batch: (real_a, real_b, is_paired_mask)
        # For simplicity, we'll assume the data loader provides pairs 
        # but we treat them as unpaired unless we want to use the 'paired' loss.
        real_a, real_b = batch
        opt_g, opt_d = self.optimizers()

        # ─── G Step ───
        # Identity
        loss_id_a = F.l1_loss(self.g_ba(real_a), real_a) * self.lambda_id
        loss_id_b = F.l1_loss(self.g_ab(real_b), real_b) * self.lambda_id

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

        # Weak Supervision (Paired L1)
        # Even if we don't have perfect registration, L1 helps guide the global structure
        loss_paired = F.l1_loss(fake_b, real_b) * self.lambda_paired

        g_loss = loss_gan_ab + loss_gan_ba + loss_cycle_a + loss_cycle_b + loss_id_a + loss_id_b + loss_paired
        
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        # ─── D Step ───
        loss_d_a = (F.mse_loss(self.d_a(real_a), torch.ones_like(self.d_a(real_a))) + 
                    F.mse_loss(self.d_a(fake_a.detach()), torch.zeros_like(self.d_a(fake_a.detach())))) * 0.5
        loss_d_b = (F.mse_loss(self.d_b(real_b), torch.ones_like(self.d_b(real_b))) + 
                    F.mse_loss(self.d_b(fake_b.detach()), torch.zeros_like(self.d_b(fake_b.detach())))) * 0.5

        opt_d.zero_grad()
        self.manual_backward(loss_d_a + loss_d_b)
        opt_d.step()

        self.log_dict({
            "g_loss": g_loss,
            "d_loss": loss_d_a + loss_d_b,
            "paired_l1": loss_paired
        }, prog_bar=True)

    def configure_optimizers(self):
        params_g = itertools.chain(self.g_ab.parameters(), self.g_ba.parameters())
        params_d = itertools.chain(self.d_a.parameters(), self.d_b.parameters())
        opt_g = torch.optim.Adam(params_g, lr=1e-4, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(params_d, lr=1e-4, betas=(0.5, 0.999))
        return [opt_g, opt_d]
