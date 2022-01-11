from .style_net import build_stylenet
import numpy as np
import torch
from .base import BaseGenerator
from .. import layers
from ..build import GENERATOR_REGISTRY
from ..layers.stylegan2_layers import Conv2dLayer, StyleGAN2Block, ToRGBLayer, FullyConnectedLayer
from typing import List, Optional


def get_chsize(imsize, cnum, max_imsize, max_cnum_mul):
    n = int(np.log2(max_imsize) - np.log2(imsize))
    mul = min(2**n, max_cnum_mul)
    ch = cnum * mul
    return int(ch)


@GENERATOR_REGISTRY.register_module
class DecoderGenerator(BaseGenerator):

    def __init__(
                # See configs/segan/base.py for default values
                self,
                image_channels: int,
                imsize: List[int],
                cnum: int,
                max_cnum_mul: int,
                z_channels: int,
                semantic_nc: Optional[int],
                conv_clamp: int,
                use_norm: bool,
                style_cfg: dict,
                embed_z: bool,
                class_specific_z: bool,
                cse_nc: int,
                use_cse: bool = True,
                use_noise: bool = False,
                *args,
                **kwargs
                ) -> None:
        super().__init__(z_channels)
        if use_cse:
            semantic_nc = None
        else:
            cse_nc = None
        assert semantic_nc is None or cse_nc is None
        semantic_nc = 0 if semantic_nc is None else semantic_nc
        cse_nc = 0 if cse_nc is None else cse_nc
        semantic_nc += cse_nc
        use_w = False
        w_dim = None
        self.imsize = imsize
        self.class_specific_z = class_specific_z
        self._cnum = cnum
        self._max_cnum_mul = max_cnum_mul
        self._min_fmap_resolution = 8
        self._image_channels = image_channels
        self._max_imsize = max(imsize)
        self._z_channels = z_channels
        self.semantic_nc = semantic_nc
        self.embed_z = embed_z
        n_levels = int(np.log2(self._max_imsize) - np.log2(self._min_fmap_resolution))+1

        # initialize decoder
        decoder_layers = []

        max_ch = 0
        feature_sizes_dec = []
        for i in range(n_levels):
            resolution = [x//2**(n_levels-1-i) for x in imsize]
            in_ch = get_chsize(max(resolution)//2, cnum, self._max_imsize, max_cnum_mul)
            max_ch = max(max_ch, in_ch)
            out_ch = get_chsize(max(resolution), cnum, self._max_imsize, max_cnum_mul)
            if i == 0:  # first (lowest) block
                in_ch = get_chsize(max(resolution), cnum, self._max_imsize, max_cnum_mul)
            up = 1
            if i != n_levels - 1:
                up = 2
            block = StyleGAN2Block(
                in_ch, out_ch, resolution, w_dim, up=up, use_w=use_w, conv_clamp=conv_clamp, use_noise=use_noise,
                use_norm=use_norm)
            decoder_layers.append(block)
            feature_sizes_dec.append([in_ch, *resolution])
            feature_sizes_dec.append([out_ch, *resolution])
        res = [x//2**(n_levels-1) for x in imsize]
        feature_sizes_dec.insert(0, [max_ch, *res])
        self.x = torch.nn.Parameter(torch.randn((1, max_ch, *res)))
        self.first = Conv2dLayer(max_ch, max_ch, None, res, conv_clamp=conv_clamp, use_norm=use_norm)
        if self.embed_z:
            self.z_projector = FullyConnectedLayer(z_channels, max_ch)
        self.decoder = layers.Sequential(*decoder_layers)
        self.to_rgb = ToRGBLayer(cnum, image_channels, w_dim, conv_clamp=conv_clamp, use_w=use_w)
        self.style_net = build_stylenet(style_cfg,
            feature_sizes_dec=feature_sizes_dec,
            cse_nc=semantic_nc,
            semantic_nc=semantic_nc,
            z_channels=z_channels)

    def get_z(self, x):
        if self.class_specific_z:
            z = torch.randn(
                (x.shape[0], self.semantic_nc, self.z_channels),
                device=x.device, dtype=x.dtype)
        else:
            z = torch.randn((x.shape[0], self.z_channels), device=x.device, dtype=x.dtype)
        return z

    def forward(self, condition, mask, z=None, w=None, semantic_mask=None, E_mask=None, vertices=None, update_ema=False, E=None, modulation_parameters=None,**kwargs):
        if z is None:
            z = self.get_z(condition)
        if modulation_parameters is None:
            modulation_parameters = self.style_net(
                semantic_mask=semantic_mask, z=z, vertices=vertices, w=w, E_mask=E_mask, update_ema=update_ema, E=E)
        batch = {"mask": mask, "modulation_params": iter(modulation_parameters)}
        batch["x"] = self.x.repeat(condition.shape[0], 1, 1, 1)
        if self.embed_z:
            z = self.z_projector(z).view(z.shape[0], -1, 1, 1)#.repeat(1, 1, *batch["x"].shape[-2:])
            batch["x"] = batch["x"] + z 
        batch["x"] = self.first(batch["x"], **next(batch["modulation_params"]))

        batch = self.decoder(batch)
        x = self.to_rgb(batch)["x"]
        return {"img": x}
