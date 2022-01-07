import torch
import numpy as np
import torch.nn as nn
from fba import layers
from ..layers import Conv2dLayer, DiscriminatorEpilogue, StyleGAN2Block
from ..build import DISCRIMINATOR_REGISTRY


@DISCRIMINATOR_REGISTRY.register_module
class StyleDiscriminator(layers.Module):

    def __init__(
            self,
            cnum: int,
            max_cnum_mul: int,
            imsize,
            min_fmap_resolution: int,
            image_channels: int,
            input_condition: bool,
            semantic_nc: int,
            semantic_input_mode: str,
            conv_clamp: int,
            *args,
            **kwargs):
        super().__init__()
        self._max_imsize = max(imsize)
        self._cnum = cnum
        self._max_cnum_mul = max_cnum_mul
        self._min_fmap_resolution = min_fmap_resolution
        self._input_condition = input_condition
        self.semantic_input_mode = semantic_input_mode
        self.layers = nn.ModuleList()
        semantic_nc = 0 if semantic_nc is None else semantic_nc
        self.from_rgb = Conv2dLayer(
            image_channels*2 + 2*input_condition + semantic_nc*(semantic_input_mode == "at_input"),
            self.get_chsize(self._max_imsize), None, imsize, 1, conv_clamp=conv_clamp
        )
        n_levels = int(np.log2(self._max_imsize) - np.log2(min_fmap_resolution))+1
        for i in range(n_levels):
            resolution = [x//2**i for x in imsize]
            in_ch = self.get_chsize(max(resolution))
            out_ch = self.get_chsize(max(max(resolution)//2, min_fmap_resolution))
            if semantic_input_mode == "progressive_input":
                self.layers.add_module(f"sematic_input{'x'.join([str(_) for _ in resolution])}", layers.SemanticCat())
                in_ch += semantic_nc
            down = 2
            if i == n_levels-1:
                down = 1
            block = StyleGAN2Block(
                in_ch, out_ch, resolution=resolution, down=down, conv_clamp=conv_clamp
            )
            self.layers.append(block)
        self.output_layer = DiscriminatorEpilogue(
            out_ch, resolution, conv_clamp=conv_clamp)

    def forward(self, img, condition, mask, semantic_mask=None, **kwargs):
        to_cat = [img]
        if self.semantic_input_mode == "at_input":
            to_cat.append(semantic_mask)
        if self._input_condition:
            to_cat.extend([condition, mask, 1-mask])
        x = torch.cat(to_cat, dim=1)
        x = self.from_rgb(x)
        batch = {"x": x, "semantic_mask": semantic_mask, "mask": None}
        for layer in self.layers:
            batch = layer(batch)
        batch = self.output_layer(batch)
        x = batch["x"]
        return dict(score=x)

    def get_chsize(self, imsize):
        n = np.log2(self._max_imsize) - np.log2(imsize)
        mul = min(2 ** n, self._max_cnum_mul)
        ch = self._cnum * mul
        return int(ch)
