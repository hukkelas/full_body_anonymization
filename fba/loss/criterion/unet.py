import torch
from .build import CRITERION_REGISTRY
from fba import utils
import torch.nn.functional as F


def get_class_balancing(semantic_mask):
    class_occurence = torch.sum(semantic_mask, dim=(0, 2, 3))
    num_of_classes = (class_occurence > 0).sum()
    coefficients = torch.reciprocal(class_occurence) * torch.numel(semantic_mask) / (num_of_classes * semantic_mask.shape[1])
    integers = torch.argmax(semantic_mask, dim=1, keepdim=True)
    weight_map = coefficients[integers]
    return weight_map


def get_target_tensor(semantic_mask, mask, is_real):
    N, semantic_nc, H, W = semantic_mask.shape
    if is_real:
        isfake_class = torch.zeros_like(mask) # All targets keeps their semantic label
    else:
        isfake_class = (1-mask) # If not real, predict fake class for all pixels masked out
        semantic_mask = semantic_mask * mask # Remove class label 
    target = torch.cat((isfake_class, semantic_mask), dim=1)
    return target


@CRITERION_REGISTRY.register_module
def segmentation_g_loss(D_out_fake, semantic_mask, seg_weight, **kwargs):
    weight = get_class_balancing(semantic_mask)
    assert weight.shape[1] == 1
    segmentation_loss = weight[:, 0] * F.cross_entropy(D_out_fake["segmentation"][:, :-1], semantic_mask.argmax(dim=1), reduction="none")
    ns_loss = torch.nn.functional.softplus(-D_out_fake["score"]).view(-1)
    total_loss = segmentation_loss.mean(dim=[1, 2]) * seg_weight + ns_loss
    to_log = dict(
        fake_segmentation=segmentation_loss.mean(),
        g_loss=ns_loss.mean(),
    )
    return total_loss, to_log


@CRITERION_REGISTRY.register_module
def segmentation_d_loss(D_out_real, D_out_fake, semantic_mask, seg_weight, **kwargs):
    weight = get_class_balancing(semantic_mask)
    segmentation_loss = weight[:, 0] * F.cross_entropy(D_out_real["segmentation"][:, :-1], semantic_mask.argmax(dim=1), reduction="none")
    ns_loss = torch.nn.functional.softplus(-D_out_real["score"]) \
        + torch.nn.functional.softplus(D_out_fake["score"])
    total_loss = seg_weight*segmentation_loss.mean(dim=[1, 2]) + ns_loss.view(-1)
    to_log = dict(
        real_segmentation=segmentation_loss.mean(),
        d_loss=ns_loss.mean(),
    )
    return total_loss,  to_log


@CRITERION_REGISTRY.register_module
def oasis_g_loss(D_out_fake, semantic_mask, mask, **kwargs):
    weight = get_class_balancing(semantic_mask)
    target = get_target_tensor(semantic_mask, mask, True).argmax(dim=1)
    loss = weight[:, 0] * F.cross_entropy(D_out_fake["segmentation"], target, reduction="none")
    to_log = dict(g_loss=loss.mean())
    return loss.mean(dim=[1, 2]), to_log


@CRITERION_REGISTRY.register_module
def oasis_d_loss(D_out_real, D_out_fake, semantic_mask, mask, **kwargs):
    weight = get_class_balancing(semantic_mask)
    target_real = get_target_tensor(semantic_mask, mask, True).argmax(dim=1)
    target_fake = get_target_tensor(semantic_mask, mask, False)
    loss_real = weight[:, 0] * F.cross_entropy(D_out_real["segmentation"], target_real, reduction="none")
    weight_fake = get_class_balancing(target_fake)
    loss_fake = weight_fake[:, 0] * F.cross_entropy(D_out_fake["segmentation"], target_fake.argmax(dim=1), reduction="none")

    total_loss = (loss_real + loss_fake).mean(dim=[1, 2])
    to_log = {
        "unet_real": loss_real.mean(),
        "unet_fake": loss_fake.mean()
    }
    return total_loss,  to_log


@CRITERION_REGISTRY.register_module
def fpn_g_loss(D_out_fake, semantic_mask, mask, segmentation_weight: float, **kwargs):
    weight = get_class_balancing(semantic_mask)
    target = get_target_tensor(semantic_mask, mask, True).argmax(dim=1)
    segmentation_loss = (weight[:, 0] * F.cross_entropy(D_out_fake["segmentation"], target, reduction="none")).mean(dim=[1, 2])

    ns_loss = torch.nn.functional.softplus(-D_out_fake["score"]).view(-1)
    total_loss = ns_loss + segmentation_loss * segmentation_weight
    to_log = dict(
        g_cross_entropy=segmentation_loss.mean(),
        g_loss=ns_loss.mean()
    )
    return total_loss, to_log


@CRITERION_REGISTRY.register_module
def fpn_d_loss(
        D_out_real, D_out_fake, semantic_mask, mask, segmentation_weight: float,
        lambda_real: float, lambda_fake: float, **kwargs):
    weight = get_class_balancing(semantic_mask)
    target_real = get_target_tensor(semantic_mask, mask, True).argmax(dim=1)
    target_fake = get_target_tensor(semantic_mask, mask, False)
    loss_real = weight[:, 0] * F.cross_entropy(D_out_real["segmentation"], target_real, reduction="none")
    weight_fake = get_class_balancing(target_fake)
    loss_fake = weight_fake[:, 0] * F.cross_entropy(D_out_fake["segmentation"], target_fake.argmax(dim=1), reduction="none")
    d_loss = torch.nn.functional.softplus(-D_out_real["score"]) \
        + torch.nn.functional.softplus(D_out_fake["score"])

    segmentation_loss = (loss_real*lambda_real + loss_fake*lambda_fake).mean(dim=[1, 2])
    total_loss = segmentation_loss*segmentation_weight + d_loss.view(-1)
    to_log = dict(
        real_cross_entropy=loss_real.mean(),
        fake_cross_entropy=loss_fake.mean(),
        d_loss=d_loss,
        real_segm_score=D_out_real["segmentation"][:, 0].mean(),
        fake_segm_score=D_out_fake["segmentation"][:, 0].mean()
    )
    return total_loss, to_log
