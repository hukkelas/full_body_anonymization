import torch
from .build import CRITERION_REGISTRY


@CRITERION_REGISTRY.register_module
def wgan_d_loss(D_out_real, D_out_fake, weight: float, **kwargs):
    wasserstein_distance = D_out_real["score"] - D_out_fake["score"]
    to_log = {"wasserstein_distance": wasserstein_distance.detach().mean()}
    return (-wasserstein_distance).view(-1) * weight, to_log


@CRITERION_REGISTRY.register_module
def wgan_g_loss(D_out_fake, weight: float, **kwargs):
    g_loss = (- D_out_fake["score"])
    to_log = {"g_loss": g_loss.mean()}
    return g_loss.view(-1) * weight, to_log


@CRITERION_REGISTRY.register_module
def nsgan_d_loss(D_out_real, D_out_fake, weight: float, **kwargs):
    """
        Non-saturating criterion from Goodfellow et al. 2014
    """
    d_loss = torch.nn.functional.softplus(-D_out_real["score"]) \
        + torch.nn.functional.softplus(D_out_fake["score"])
    wasserstein_distance = (D_out_real["score"] - D_out_fake["score"]).squeeze()
    to_log = {
        "wasserstein_distance": wasserstein_distance.mean(),
        "d_loss": d_loss.mean()
    }
    return d_loss.view(-1) * weight, to_log


@CRITERION_REGISTRY.register_module
def nsgan_g_loss(D_out_fake, weight: float, **kwargs):
    """
        Non-saturating criterion from Goodfellow et al. 2014
    """
    g_loss = torch.nn.functional.softplus(-D_out_fake["score"])
    to_log = {"g_loss": g_loss.mean()}
    return g_loss.view(-1) * weight, to_log
