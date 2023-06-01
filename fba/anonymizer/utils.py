import numpy as np
import torch
from skimage.morphology import disk
from torchvision.transforms.functional import resize, InterpolationMode
from functools import lru_cache
from fba import utils


@lru_cache(maxsize=200)
def get_kernel(n: int):
    kernel = disk(n, dtype=bool)
    return utils.to_cuda(torch.from_numpy(kernel).bool())


def transform_embedding(E: torch.Tensor, S: torch.Tensor, exp_bbox, E_bbox, target_imshape):
    """
        Transforms the detected embedding/mask directly to the target image shape
    """

    C, HE, WE = E.shape
    assert E_bbox[0] >= exp_bbox[0], (E_bbox, exp_bbox)
    assert E_bbox[2] >= exp_bbox[0]
    assert E_bbox[1] >= exp_bbox[1]
    assert E_bbox[3] >= exp_bbox[1]
    assert E_bbox[2] <= exp_bbox[2]
    assert E_bbox[3] <= exp_bbox[3]

    x0 = int(np.round((E_bbox[0] - exp_bbox[0]) / (exp_bbox[2] - exp_bbox[0]) * target_imshape[1]))
    x1 = int(np.round((E_bbox[2] - exp_bbox[0]) / (exp_bbox[2] - exp_bbox[0]) * target_imshape[1]))
    y0 = int(np.round((E_bbox[1] - exp_bbox[1]) / (exp_bbox[3] - exp_bbox[1]) * target_imshape[0]))
    y1 = int(np.round((E_bbox[3] - exp_bbox[1]) / (exp_bbox[3] - exp_bbox[1]) * target_imshape[0]))
    new_E = torch.zeros((C, *target_imshape), device=E.device, dtype=torch.float32)
    new_S = torch.zeros((target_imshape), device=S.device, dtype=torch.bool)

    E = resize(E, (y1-y0, x1-x0), antialias=True, interpolation=InterpolationMode.BILINEAR)
    new_E[:, y0:y1, x0:x1] = E
    S = resize(S[None].float(), (y1-y0, x1-x0), antialias=True, interpolation=InterpolationMode.BILINEAR)[0] > 0
    new_S[y0:y1, x0:x1] = S
    return new_E, new_S


def pairwise_mask_iou(mask1: torch.Tensor, mask2: torch.Tensor):
    """
        mask: shape [N, H, W]
    """
    assert len(mask1.shape) == 3
    assert len(mask2.shape) == 3
    assert mask1.device == mask2.device, (mask1.device, mask2.device)
    assert mask2.dtype == mask2.dtype
    assert mask1.dtype == torch.bool
    assert mask1.shape[1:] == mask2.shape[1:]
    N1, H1, W1 = mask1.shape
    N2, H2, W2 = mask2.shape
    iou = torch.zeros((N1, N2), dtype=torch.float32)
    for i in range(N1):
        cur = mask1[i:i+1]
        inter = torch.logical_and(cur, mask2).flatten(start_dim=1).float().sum(dim=1).cpu()
        union = torch.logical_or(cur, mask2).flatten(start_dim=1).float().sum(dim=1).cpu()
        iou[i] = inter / union
    return iou


def find_best_matches(mask1: torch.Tensor, mask2: torch.Tensor, iou_threshold: float):
    N1 = mask1.shape[0]
    N2 = mask2.shape[0]
    ious = pairwise_mask_iou(mask1, mask2).cpu().numpy()
    indices = np.array([idx for idx, iou in np.ndenumerate(ious)])
    ious = ious.flatten()
    mask = ious >= iou_threshold
    ious = ious[mask]
    indices = indices[mask]

    # do not sort by iou to keep ordering of mask rcnn / cse sorting.
    taken1 = np.zeros((N1), dtype=bool)
    taken2 = np.zeros((N2), dtype=bool)
    matches = []
    for i, j in indices:
        if taken1[i].any() or taken2[j].any():
            continue
        matches.append((i, j))
        taken1[i] = True
        taken2[j] = True
    return matches
