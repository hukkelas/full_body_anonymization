
import collections
import numpy as np
import torch
from fba import utils
from fba.infer import build_trained_generator
from fba.infer import TruncationStrategy, sample_from_G
from fba.build import build_generator
from .cse import CSEDetector
import torchvision.transforms.functional as F
from .post_process import process_cse_detections


def build_anonymizer(cfg, detection_score_threshold, **kwargs):
    detector = CSEDetector(score_thres=detection_score_threshold)
    if hasattr(cfg, "dummy_anonymizer") and cfg.dummy_anonymizer:
        generator = build_generator(cfg)
    else:
        generator, _ = build_trained_generator(cfg)
    return Anonymizer(detector, generator, **kwargs)


def paste_im(target_im, im, mask_filled, mask, exp_bbox):
    x0, y0, x1, y1 = exp_bbox
    H, W = target_im.shape[-2:]
    mask = mask[max(-y0, 0):im.shape[1] - max(y1 - H, 0), max(-x0, 0):im.shape[2] - max(x1 - W, 0)]
    im = im[:, max(-y0, 0):im.shape[1] - max(y1 - H, 0), max(-x0, 0):im.shape[2] - max(x1 - W, 0)]
    mask_available = mask_filled[max(0, y0):min(y1, H), max(0, x0):min(x1, W)].logical_not()
    to_fill = mask_available
    blur_background = False
    if not blur_background:
        to_fill = mask_available.logical_and(mask)
    to_fill = to_fill[None].repeat(3, 1, 1)
    assert target_im[:, max(0, y0):min(y1, H),
        max(0, x0):min(x1, W)].shape == im.shape, (target_im[:, max(0, y0):min(y1, H),
        max(0, x0):min(x1, W)].shape, im.shape)
    target_im[:, max(0, y0):min(y1, H),
        max(0, x0):min(x1, W)][to_fill] = im[to_fill]
    mask_filled[max(0, y0):min(y1, H),
        max(0, x0):min(x1, W)][mask] = 1
    return target_im


def stitch_image(orig_im, generated_images, dilated_mask, E_mask, expanded_boxes):
    im = orig_im.clone()
    mask_filled = torch.zeros(im.shape[-2:], dtype=torch.bool, device=im.device)
    borders = []
    resized_bodies = []
    for body_idx, body in enumerate(generated_images):
        orig_bbox = expanded_boxes[body_idx]
        e_mask = E_mask[body_idx]
        d_mask = dilated_mask[body_idx]
        assert len(e_mask.shape) == 2, e_mask.shape
        target_shape = (orig_bbox[3] - orig_bbox[1], orig_bbox[2] - orig_bbox[0])
        body = F.resize(body[None], target_shape, antialias=True)[0]
        e_mask = F.resize(e_mask[None, None], target_shape, antialias=True)[0, 0] > 0
        d_mask = F.resize(d_mask[None, None], target_shape, antialias=True)[0, 0] > 0
        border = e_mask.logical_xor(d_mask)
        resized_bodies.append(body)
        borders.append(border)
        im = paste_im(im, body, mask_filled, e_mask, orig_bbox)
    for body_idx, body in enumerate(generated_images):
        orig_bbox = expanded_boxes[body_idx]
        border = borders[body_idx]
        target_shape = (orig_bbox[3] - orig_bbox[1], orig_bbox[2] - orig_bbox[0])
        body = resized_bodies[body_idx]
        im = paste_im(im, body, mask_filled, border, orig_bbox)
    
    return im


class Anonymizer:

    def __init__(
            self,
            detector,
            generator) -> None:
        self.detector = detector
        self.generator = generator
        self.z = None
        self.embed_map = self.detector.mesh_vertex_embeddings["smpl_27554"]
        self.batch_size = 2

    @torch.no_grad()
    def forward(self, im: torch.Tensor, truncation_value: float = None) -> np.ndarray:
        assert im.dtype == torch.uint8
        im = utils.to_cuda(im)
        detections = self.detector(im)
        if detections is None:
            return im
        post_process_cfg=dict(
            target_imsize=(288, 160),
            exp_bbox_cfg=dict(percentage_background=0.3, axis_minimum_expansion=.1),
            exp_bbox_filter=dict(minimum_area=32*32, min_bbox_ratio_inside=0, aspect_ratio_range=[0, 99999]),
        )
        detections = list(process_cse_detections(
            im, **detections, **post_process_cfg
        ))
        boxes = []
        anonymized_bodies = []
        masked_areas = []
        E_masks = []
        cur_batch = collections.defaultdict(list)
        for idx, instance in enumerate(detections):
            boxes.append(instance["exp_bbox"])
            cur_batch["mask"].append(instance["mask"].float())
            cur_batch["border"].append(instance["border"].float())
            cur_batch["img"].append(instance["im"])
            cur_batch["E_mask"].append(instance["E_mask"])
            cur_batch["vertices"].append(instance["vertices"][None])
            if idx == len(detections) - 1 or ((idx+1) % self.batch_size == 0):
                batch = {k: torch.cat(v, dim=0) for k, v in cur_batch.items()}
                orig_shape = batch["img"].shape[2:]
                batch["condition"] = batch["img"] * batch["mask"]
                batch["embedding"] = self.embed_map[batch["vertices"]].permute(0, 3, 1, 2) * batch["E_mask"]
                anonymized = sample_from_G(batch, self.generator, TruncationStrategy.W_INTERPOLATE, truncation_value)["img"]
                anonymized = F.resize(anonymized, size=orig_shape, antialias=True)
                anonymized = utils.denormalize_img(anonymized).mul(255).byte()
                anonymized_bodies.extend(anonymized)
                masked_areas.extend((batch["mask"].bool().logical_not()).squeeze(dim=1))
                E_masks.extend(batch["E_mask"].squeeze(dim=1))
                cur_batch = collections.defaultdict(list)
        im = stitch_image(im, anonymized_bodies, masked_areas, E_masks, boxes)
        return im

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
