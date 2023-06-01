from torch_utils.ops import upfirdn2d
import torch
import numpy as np
import torch.nn as nn
from .. import layers
from ..layers.stylegan2_layers import Conv2dLayer, DiscriminatorEpilogue, StyleGAN2Block
from ..build import DISCRIMINATOR_REGISTRY


@DISCRIMINATOR_REGISTRY.register_module
class FPNDiscriminator(layers.Module):

    def __init__(
            self,
            cnum: int,
            max_cnum_mul: int,
            imsize,
            min_fmap_resolution: int,
            image_channels: int,
            conv_clamp: int,
            input_cse: bool,
            cse_nc: int,
            output_fpn: bool,
            input_condition=True,
            *args,
            **kwargs):
        super().__init__()
        self._max_imsize = max(imsize)
        self._cnum = cnum
        self._max_cnum_mul = max_cnum_mul
        self._min_fmap_resolution = min_fmap_resolution
        self.input_cse = input_cse
        self.output_fpn = output_fpn
        self.input_condition = input_condition
        self.layers = nn.ModuleList()

        out_ch = self.get_chsize(self._max_imsize)
        self.from_rgb = StyleGAN2Block(
            image_channels + (image_channels+1)*self.input_condition,
            out_ch, imsize, architecture="orig", conv_clamp=conv_clamp
        )
        n_levels = int(np.log2(self._max_imsize) - np.log2(min_fmap_resolution))+1
        if self.output_fpn:
            self.fpn_out = nn.ModuleList()
            self.fpn_up = nn.ModuleList()
            self.output_seg_layer = Conv2dLayer(
                128, cse_nc, None, kernel_size=1, activation="linear")

        for i in range(n_levels):
            resolution = [x//2**i for x in imsize]
            in_ch = self.get_chsize(max(resolution))
            out_ch = self.get_chsize(max(max(resolution)//2, min_fmap_resolution))

            if i != n_levels - 1 and output_fpn:
                fpn_up_in_ = 128 if i != n_levels - 2 else in_ch
                up = 2 if i != 0 else 1
                fpn_up = Conv2dLayer(fpn_up_in_, 128, resolution, kernel_size=1, activation="linear", up=up, conv_clamp=conv_clamp)
                self.fpn_up.append(fpn_up)
                fpn_conv = Conv2dLayer(in_ch, 128, resolution, kernel_size=1, activation="linear", conv_clamp=conv_clamp)
                self.fpn_out.append(fpn_conv)


            down = 2
            if i == 0:
                down = 1
            block = StyleGAN2Block(
                in_ch, out_ch, resolution=resolution, down=down, conv_clamp=conv_clamp
            )
            self.layers.append(block)
        self.output_layer = DiscriminatorEpilogue(
            out_ch, resolution, conv_clamp=conv_clamp)

        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1, 3, 3, 1]))

    def forward(self, img, condition, mask, semantic_mask=None, embedding=None, **kwargs):
        to_cat = [img]
        if self.input_condition:
            to_cat.extend([condition, mask])
        x = torch.cat(to_cat, dim=1)
        batch = {"x": x, "mask": None}
        batch = self.from_rgb(batch)
        if self.output_fpn:
            fpn_skips = [self.fpn_out[0](batch["x"])]

        for i, layer in enumerate(self.layers):
            batch = layer(batch)
            if not self.output_fpn:
                continue
            if i < len(self.layers)-2:
                fpn_skips.append(
                    self.fpn_out[i+1](batch["x"], gain=np.sqrt(.5))
                )
            elif i == len(self.layers) - 2:
                fpn_skips.append(batch["x"])
        
        
        batch = self.output_layer(batch)
        if not self.output_fpn:
            return dict(score=batch["x"])
        fpn_skips.reverse()
        E = fpn_skips[0]
        for i in range(len(self.fpn_up)):
            E = self.fpn_up[-i-1](E, gain=np.sqrt(0.5))
            E = (E + fpn_skips[i+1])
        E = self.output_seg_layer(E)
        return dict(score=batch["x"], E=E)

    def get_chsize(self, imsize):
        n = int(np.log2(self._max_imsize) - np.log2(imsize))
        mul = min(2 ** n, self._max_cnum_mul)
        ch = self._cnum * mul
        return int(ch)
