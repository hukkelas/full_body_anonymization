import torch
from .build import CRITERION_REGISTRY
import torch.nn.functional as F


def masked_l1(x, target, M):
    return (F.smooth_l1_loss(x*M, target*M, reduction="none").sum(dim=[1, 2, 3]) / M.sum(dim=[1, 2, 3]))


@CRITERION_REGISTRY.register_module
def fpn_cse_g_loss(D_out_fake, mask, border, embedding, l1_weight: float, **kwargs):
    mask_plus_b = mask + border
    E = embedding
    assert D_out_fake["E"].shape == E.shape, (E.shape, D_out_fake["E"].shape)
    l1 = masked_l1(D_out_fake["E"], E, 1-mask_plus_b)
    ns_loss = torch.nn.functional.softplus(-D_out_fake["score"]).view(-1)
    total_loss = ns_loss + l1 * l1_weight

    to_log = dict(
        g_embedding_l1=l1.mean(),
        g_loss=ns_loss.mean(),
    )
    return total_loss, to_log


@CRITERION_REGISTRY.register_module
def fpn_cse_d_loss(
        D_out_real, D_out_fake, mask, border, embedding, l1_weight: float,
        lambda_real: float, lambda_fake: float, **kwargs):
    mask_plus_b = mask + border
    l1_real = masked_l1(D_out_real["E"], embedding, 1-mask_plus_b)
    l1_fake = masked_l1(D_out_fake["E"], embedding, 1-mask_plus_b)
    d_loss = torch.nn.functional.softplus(-D_out_real["score"]) \
        + torch.nn.functional.softplus(D_out_fake["score"])

    l1_loss = (l1_real*lambda_real + l1_fake*lambda_fake)
    total_loss = l1_loss*l1_weight + d_loss.view(-1)

    to_log = dict(
        D_l1_real=l1_real.mean(),
        D_l1_fake=l1_fake.mean(),
        d_loss=d_loss,
    )
    return total_loss, to_log


@CRITERION_REGISTRY.register_module
def uncond_fpn_cse_g_loss(D_out_fake, E_mask, embedding, l1_weight: float, **kwargs):
    mask = 1 - E_mask
    border = torch.zeros_like(mask)
    return fpn_cse_g_loss(D_out_fake, mask, border, embedding, l1_weight)


@CRITERION_REGISTRY.register_module
def uncond_fpn_cse_d_loss(
        D_out_real, D_out_fake, E_mask, embedding, l1_weight: float,
        lambda_real: float, lambda_fake: float, **kwargs):
    mask = 1 - E_mask
    border = torch.zeros_like(mask)
    return fpn_cse_d_loss(D_out_real, D_out_fake, mask, border, embedding, l1_weight, lambda_real, lambda_fake)

