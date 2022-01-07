import torch
import torchvision
from fba import utils
import numpy as np
from .bbox_utils import get_surrounding_bbox, include_box
from .ops import binary_dilation
from .utils import get_kernel


def dilate_S(mask):
    n_pixels = mask.long().sum().item()
    dilation_amount = int(np.ceil(8 * np.sqrt(n_pixels/(256**2))))
    dilation_amount = max(dilation_amount, 1)
    selem = get_kernel(dilation_amount)
    old = mask
    mask = binary_dilation(mask.bool(), selem).float()
    border = mask - old
    return mask, border


def transform_EM(E, mask, bbox, E_bbox_XYWH):
    assert E.shape[1:] == mask.shape[:2], (E.shape, mask.shape)
    E_x0, E_y0, E_x1, E_y1 = E_bbox_XYWH

    pad = np.array([[E_y0 - bbox[1], bbox[3] - E_y1], [E_x0 - bbox[0], bbox[2] - E_x1]])
    E = E[:, max(-pad[0,0], 0):E.shape[1] + min(pad[0,1], 0), max(-pad[1,0], 0):E.shape[2] + min(pad[1,1], 0)]
    mask = mask[max(-pad[0,0], 0):mask.shape[0] + min(pad[0,1], 0), max(-pad[1,0], 0):mask.shape[1] + min(pad[1,1], 0)]
    pad[pad < 0] = 0
    # Torch pad starts from last dimension and goes backward. Dimension is CxHxW
    pad = [pad[1, 0], pad[1,1], pad[0, 0], pad[0, 1]]
    pad_E = [*pad, 0, 0]
    E = torch.nn.functional.pad(E, pad_E, mode="constant", value=0)
    mask = torch.nn.functional.pad(mask, pad, mode="constant", value=0)
    assert E.shape[1:] == (bbox[3] - bbox[1], bbox[2] - bbox[0]), (E.shape, (bbox[3] - bbox[1], bbox[2] - bbox[0]))
    assert mask.shape[:2] == (bbox[3] - bbox[1], bbox[2] - bbox[0]), (mask.shape, (bbox[3] - bbox[1], bbox[2] - bbox[0]))
    return E, mask


def cut_and_pad(im: torch.Tensor, E, mask, bbox, target_shape, E_bbox_XYWH):
    assert im.dtype == torch.uint8

    x0, y0, x1, y1 = bbox
    E, mask = transform_EM(E, mask, bbox, E_bbox_XYWH)
    new_M = torch.nn.functional.interpolate(mask[None, None].float(), target_shape, mode="nearest")
    new_M, border = dilate_S(new_M)
    new_E = torchvision.transforms.functional.resize(E[None], target_shape, antialias=True) 
    #new_E = torch.nn.functional.interpolate(E[None], target_shape, mode="bilinear", align_corners=False) 

    new_im = torch.zeros((3, y1-y0, x1-x0), dtype=torch.uint8)
    y0_t = max(0, -y0)
    y1_t = min(y1-y0, (y1-y0)-(y1-im.shape[1]))
    x0_t = max(0, -x0)
    x1_t = min(x1-x0, (x1-x0)-(x1-im.shape[2]))
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(x1, im.shape[2])
    y1 = min(y1, im.shape[1])
    new_im[:, y0_t:y1_t, x0_t:x1_t] = im[:, y0:y1, x0:x1]
    new_im = utils.to_cuda(new_im.float().div(255).mul(2).sub(1))[None]
    new_im = torchvision.transforms.functional.resize(new_im, target_shape, antialias=True)
    return new_im, new_E, new_M, border


def process_cse_detections(
        im: torch.Tensor,
        instance_segmentation: torch.Tensor, instance_embedding: torch.Tensor,
        embed_map: torch.Tensor, bbox_XYXY: torch.Tensor,
        target_imsize, exp_bbox_cfg, exp_bbox_filter, **kwargs):
    N = instance_segmentation.size(0)
    instance_segmentation = instance_segmentation.view(-1, 1, *instance_segmentation.shape[-2:]).float()
    for i in range(N):
        x0, y0, x1, y1 = bbox_XYXY[i].int().tolist()
        E_ = torch.nn.functional.interpolate(instance_embedding[i:i+1], size=(y1-y0, x1-x0), mode="bilinear", align_corners=False)
        S = torch.nn.functional.interpolate(instance_segmentation[i:i+1], size=(y1-y0, x1-x0), mode="bilinear", align_corners=False)[0] > 0
        exp_bbox, tight_bbox = get_surrounding_bbox(
            S.cpu().numpy().squeeze(), bbox_XYXY[i].cpu().numpy(), im.shape[-2:],
            percentage_background=0.3, axis_minimum_expansion=.1,
            target_aspect_ratio=288/160)
        if not include_box(
                tight_bbox, minimum_area=32*32,
                aspect_ratio_range=[.0, 99999],
                min_bbox_ratio_inside=0,
                imshape=im.shape[-2:]):
            continue
        im_cut, E_cut, S_dilated, border = cut_and_pad(
            im, E_.squeeze(), S.squeeze(),
            exp_bbox, target_imsize,
            bbox_XYXY[i].cpu().tolist())
        E_mask = S_dilated.logical_xor(border)
        vertices = utils.from_E_to_vertex(
            E_cut, E_mask.logical_not(), embed_map)
        yield dict(
            im=im_cut,
            E=E_cut,
            mask=S_dilated.bool().logical_not(),
            border=border.bool(),
            E_mask=E_mask,
            exp_bbox=exp_bbox,
            vertices=vertices,
            N=N
        )
