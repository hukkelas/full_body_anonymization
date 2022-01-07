import torch
from fba import utils
from .build import CRITERION_REGISTRY


@CRITERION_REGISTRY.register_module
def epsilon_penalty(D_out_real, weight: float):
    epsilon_penalty = D_out_real["score"].pow(2)
    to_log = dict(epsilon_penalty=epsilon_penalty.mean())
    loss = epsilon_penalty * weight
    return loss.view(-1), to_log


@CRITERION_REGISTRY.register_module
def improved_gradient_penalty(
        batch, fake_data,
        weight: float, lazy_reg_interval: int,
        scaler: torch.cuda.amp.GradScaler,
        discriminator, **kwargs):
    """
        Wasserstein gradient penalty.
        Implementation from:
            Hukkelås et al. Image Inpainting with Learnable Feature Imputation. GCPR 2020
    """
    real_data = batch["img"]
    mask = batch["mask"]
    epsilon_shape = [real_data.shape[0]] + [1] * (real_data.dim() - 1)
    epsilon = torch.rand(epsilon_shape, device=fake_data.device, dtype=fake_data.dtype)
    real_data = real_data.to(fake_data.dtype)
    x_hat = epsilon * real_data + (1 - epsilon) * fake_data.detach()
    x_hat.requires_grad = True
    with torch.cuda.amp.autocast(enabled=utils.AMP()):
        logit = utils.forward_D_fake(batch, x_hat, discriminator)["score"]

    grad = torch.autograd.grad(
        outputs=scaler.scale(logit),
        inputs=x_hat,
        grad_outputs=torch.ones_like(logit),
        create_graph=True,
        only_inputs=True,
    )[0]
    inv_scale = 1.0 / scaler.get_scale()
    grad = grad * inv_scale
    with torch.cuda.amp.autocast(utils.AMP()):
        grad = grad * (1 - mask)
        grad = grad.view(x_hat.shape[0], -1)
        grad_norm = grad.norm(p=2, dim=1)
        gradient_pen = grad_norm - 1
        gradient_pen = gradient_pen.relu()
    x_hat.requires_grad = False
    lambd_ = weight * lazy_reg_interval  # From stylegan2, lazy regularization
    return gradient_pen * lambd_, dict(gradient_penalty=gradient_pen)


@CRITERION_REGISTRY.register_module
def r1_regularization(
        batch, D_out_real, weight: float, lazy_reg_interval: int,
        scaler: torch.cuda.amp.GradScaler, mask_out: bool, **kwargs
        ):
    real_data = batch["img"]
    real_scores = D_out_real["score"]
    mask = batch["mask"]
    grad = torch.autograd.grad(
        outputs=scaler.scale(real_scores),
        inputs=real_data,
        grad_outputs=torch.ones_like(real_scores),
        create_graph=True,
        only_inputs=True,
    )[0]
    inv_scale = 1.0 / scaler.get_scale()
    grad = grad * inv_scale
    with torch.cuda.amp.autocast(utils.AMP()):
        if mask_out:
            grad = grad * (1 - mask)
        grad = grad.square().sum(dim=[1, 2, 3])
        if mask_out:
            total_pixels = real_data.shape[1] * real_data.shape[2] * real_data.shape[3]
            n_fake = (1-mask).sum(dim=[1, 2, 3])
            scaling = total_pixels / n_fake
            grad = grad * scaling
    lambd_ = weight * lazy_reg_interval / 2  # From stylegan2, lazy regularization
    return grad * lambd_, dict(r1_gradient_penalty=grad)