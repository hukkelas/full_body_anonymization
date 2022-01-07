import torch
from fba.utils import Registry, build_from_cfg
from fba import layers

TRANSFORM_REGISTRY = Registry("TRANSFORM")


def build_transforms(transform_cfgs, imsize, torch_script):
    requires_imsize = [
        "Resize", "CenterCrop", "RandomCrop",
        "DictResize"]
    transforms = []
    for tcfg in transform_cfgs:
        kwargs = {}
        if tcfg.type in requires_imsize and "size" not in tcfg:
            kwargs["size"] = imsize
        t = build_from_cfg(tcfg, TRANSFORM_REGISTRY, **kwargs)
        transforms.append(t)
    transform = layers.Sequential(*transforms)
    if torch_script:
        transform = torch.jit.script(transform)
    return transform
