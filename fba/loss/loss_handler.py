from torch_utils import misc
import torch
from .criterion import build_criterion
from fba.utils import forward_D_fake
from fba import utils


class LossHandler:

    def __init__(
            self,
            discriminator,
            generator,
            gan_criterion: dict,
            gradient_penalty: dict,
            epsilon_penalty: dict,
            feature_matching: dict,
            gaussian_kl_loss: dict,
            lazy_reg_interval: int,
            l1_loss: dict,
            scaler: torch.cuda.amp.GradScaler,
        ) -> None:
        self.gradient_step_D = 0
        self._lazy_reg_interval = lazy_reg_interval
        self.scaler = scaler
        self.discriminator = discriminator
        self.generator = generator
        self.gradient_pen = build_criterion(
            gradient_penalty.type, gradient_penalty,
            lazy_reg_interval=lazy_reg_interval, discriminator=self.discriminator,
            scaler=scaler)
        self.gan_d_loss = build_criterion(gan_criterion.type + "_d_loss", gan_criterion)
        self.gan_g_loss = build_criterion(gan_criterion.type + "_g_loss", gan_criterion)
        self.EP_loss = build_criterion(epsilon_penalty.type, epsilon_penalty)
        self.FM_loss = build_criterion(feature_matching.type, feature_matching)
        self.gaussian_kl_loss = build_criterion(gaussian_kl_loss.type, gaussian_kl_loss)
        self.l1_loss = build_criterion(l1_loss.type, l1_loss)
        self.r1_reg = "r1_regularization" in gradient_penalty.type 

    def D_loss(self, batch: dict):
        to_log = {}
        # Forward through G and D
        do_GP = self.gradient_pen is not None and self.gradient_step_D % self._lazy_reg_interval == 0
        if do_GP and self.r1_reg:
            batch["img"] = batch["img"].detach().requires_grad_(True)
        with torch.cuda.amp.autocast(enabled=utils.AMP()):
            with torch.no_grad():
                with misc.ddp_sync(self.generator, False):
                    G_fake = self.generator(**batch, update_ema=True)
            with misc.ddp_sync(self.discriminator, False):
                D_out_fake = forward_D_fake(batch, G_fake["img"], self.discriminator)
            with misc.ddp_sync(self.discriminator, True):
                D_out_real = self.discriminator(**batch)
        total_loss = 0

        # Adversarial Loss
        with torch.cuda.amp.autocast(enabled=utils.AMP()):
            if self.gan_d_loss is not None:
                with torch.autograd.profiler.record_function("D_loss_GAN"):
                    loss, log = self.gan_d_loss(D_out_real, D_out_fake, **batch)
                    to_log.update(log)
                    assert loss.shape == (batch["img"].shape[0], ), loss.shape
                    total_loss += loss

            # Discriminator feature loss
            if self.EP_loss is not None:
                loss, log = self.EP_loss(D_out_real)
                assert loss.shape == total_loss.shape
                to_log.update(log)
                total_loss += loss

        # Improved gradient penalty with lazy regularization 
        # Gradient penalty applies specialized autocast.
        if do_GP:
            with torch.autograd.profiler.record_function("D_loss_GP"):
                loss, log = self.gradient_pen(batch, fake_data=G_fake["img"], D_out_real=D_out_real)
                assert loss.shape == total_loss.shape
                total_loss += loss
                to_log.update(log)
        batch["img"] = batch["img"].detach().requires_grad_(False)
        if "score" in D_out_real:
            to_log["real_scores"] = D_out_real["score"]
            to_log["fake_scores"] = D_out_fake["score"]
        to_log = {key: item.mean().detach() for key, item in to_log.items()}
        self.gradient_step_D += 1
        return total_loss.mean(), to_log

    def G_loss(self, batch: dict):
        with torch.cuda.amp.autocast(enabled=utils.AMP()):
            to_log = {}
            # Forward through G and D
            with misc.ddp_sync(self.generator, True):
                G_fake = self.generator(**batch)
            with misc.ddp_sync(self.discriminator, False):
                D_out_real = self.discriminator(**batch)

            D_out_fake = forward_D_fake(batch, G_fake["img"], self.discriminator)
            total_loss = 0
            # Adversarial Loss
            if self.gan_g_loss is not None:
                with torch.autograd.profiler.record_function("G_loss_GAN"):
                    loss, log = self.gan_g_loss(D_out_fake, **batch)
                    to_log.update(log)
                    assert loss.shape == (batch["img"].shape[0], ), loss.shape
                    total_loss = total_loss + loss

            # Discriminator feature loss
            if self.FM_loss is not None:
                with torch.autograd.profiler.record_function("G_loss_FM"):
                    loss, log = self.FM_loss(D_out_real["features"], D_out_fake["features"])
                    assert loss.shape == total_loss.shape
                    to_log.update(log)
                    total_loss = total_loss + loss

            # KL Gaussian Loss
            if self.gaussian_kl_loss is not None:
                with torch.autograd.profiler.record_function("G_loss_KL"):
                    loss, log = self.gaussian_kl_loss(G_fake["z_mu"], G_fake["z_logvar"])
                    assert loss.shape == total_loss.shape
                    total_loss = total_loss + loss
                    to_log.update(log)
            if self.l1_loss is not None:
                with torch.autograd.profiler.record_function("l1_loss"):
                    loss, log = self.l1_loss(batch["img"], G_fake["img"], batch["mask"])
                    assert loss.shape == total_loss.shape
                    total_loss = total_loss + loss
                    to_log.update(log)
        to_log = {key: item.mean().detach() for key, item in to_log.items()}
        return total_loss.mean(), to_log
