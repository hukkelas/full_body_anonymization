import torch
import torch.nn as nn
from fba.build import GENERATOR_REGISTRY


@GENERATOR_REGISTRY.register_module()
class PixelationGenerator(nn.Module):

    def __init__(self, pixelation_size, imsize, **kwargs):
        super().__init__()
        self.pixelation_size = pixelation_size
        self.z_channels = 0
        self.latent_space=None

    def forward(self, img, condition, mask, **kwargs):
        old_shape = img.shape[-2:]
        img = nn.functional.interpolate(img, size=(self.pixelation_size, self.pixelation_size), mode="bilinear")
        img = nn.functional.interpolate(img, size=old_shape, mode="bilinear")
        out = img*(1-mask) + condition*mask
        return {"img": out}



@GENERATOR_REGISTRY.register_module()
class MaskOutGenerator(nn.Module):

    def __init__(self, noise: str, imsize, **kwargs):
        super().__init__()
        self.noise = noise
        self.imsize = imsize
        self.z_channels = 0
        assert self.noise in ["rand", "constant"]
        self.latent_space=None

    def forward(self, img, condition, mask, **kwargs):
        
        if self.noise == "constant":
            img = torch.zeros_like(img)
        elif self.noise == "rand":
            img = torch.rand_like(img)
        out = img*(1-mask) + condition*mask
        return {"img": out}
