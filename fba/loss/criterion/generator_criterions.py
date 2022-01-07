import torch
from fba import utils
from .build import CRITERION_REGISTRY


@CRITERION_REGISTRY.register_module
def masked_l1_loss(reals, fakes, mask, weight: float):
    l1_loss = torch.abs((reals - fakes) * (1 - mask)).view(reals.shape[0], -1)
    denom = (1 - mask).view(reals.shape[0], -1).sum(dim=1)
    l1_loss = (l1_loss.sum(dim=1) / denom)
    to_log = {"l1_loss": l1_loss.mean()}
    return l1_loss.view(-1) * weight, to_log


@CRITERION_REGISTRY.register_module
def l1_loss(reals, fakes, mask, weight: float, *args, **kwargs):
    l1_loss = torch.abs((reals - fakes)).view(reals.shape[0], -1).mean(dim=1)
    to_log = {"l1_loss": l1_loss.mean()}
    return l1_loss.view(-1) * weight, to_log


def norm(x, ord):
    x = x.view(x.shape[0], -1)
    if ord == "l1":
        return torch.abs(x).mean(dim=1)
    if "ord" == "l2":
        return (x ** 2).mean(dim=1)
    raise ValueError("Unsupported norm:", ord)


@CRITERION_REGISTRY.register_module
def discriminator_feature_matching_loss(
        features_real, features_fake,
        weight: float, resolutions: list, ord: str):
    to_log = {}
    to_backward = 0
    for res in resolutions:
        loss = norm((features_real[res].detach() - features_fake[res]), ord=ord)
        to_log[f"d_feature_match/{res}"] = loss.mean()
        to_backward += loss * weight
    return to_backward, to_log


@CRITERION_REGISTRY.register_module
def gaussian_kl_loss(z_mu, z_logvar, weight: float):
    kl_loss = utils.gaussian_kl(z_mu, z_logvar)
    to_log = dict(
        kl_divergence=kl_loss.mean(), z_mean=z_mu.mean(),
        z_logvar=z_logvar.mean())
    return kl_loss * weight, to_log
