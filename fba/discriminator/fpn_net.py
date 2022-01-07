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
            input_condition: bool,
            semantic_nc: int,
            semantic_input_mode: str,
            conv_clamp: int,
            input_cse: bool,
            cse_nc: int,
            pred_only_cse: bool = False,
            pred_only_semantic: bool = False,
            *args,
            **kwargs):
        super().__init__()
        if pred_only_cse:
            semantic_nc = None
        if pred_only_semantic:
            cse_nc = None
        self.pred_only_cse = pred_only_cse
        self.pred_only_semantic = pred_only_semantic
        assert semantic_nc is None or cse_nc is None
        semantic_nc = 0 if semantic_nc is None else semantic_nc
        cse_nc = 0 if cse_nc is None else cse_nc
        semantic_nc += cse_nc
        self._max_imsize = max(imsize)
        self._cnum = cnum
        self._max_cnum_mul = max_cnum_mul
        self._min_fmap_resolution = min_fmap_resolution
        self._input_condition = input_condition
        self.semantic_input_mode = semantic_input_mode
        self.input_cse = input_cse
        self.layers = nn.ModuleList()

        out_ch = self.get_chsize(self._max_imsize)
        self.from_rgb = StyleGAN2Block(
            image_channels + input_condition*(image_channels+1) +
            semantic_nc*(semantic_input_mode == "at_input") + input_cse*cse_nc,
            out_ch, imsize, None, architecture="orig", conv_clamp=conv_clamp
        )
        self.output_seg_layer = Conv2dLayer(
            semantic_nc, semantic_nc+1*(cse_nc==0), None, None, kernel_size=1, activation="linear")
        n_levels = int(np.log2(self._max_imsize) - np.log2(min_fmap_resolution))+1
        self.fpn_out = nn.ModuleList()
        self.fpn_up = nn.ModuleList()
        for i in range(n_levels):
            resolution = [x//2**i for x in imsize]
            in_ch = self.get_chsize(max(resolution))
            out_ch = self.get_chsize(max(max(resolution)//2, min_fmap_resolution))

            if i != n_levels - 1:
                fpn_up_in_ = semantic_nc if i != n_levels - 2 else in_ch
                up = 2 if i != 0 else 1
                fpn_up = Conv2dLayer(fpn_up_in_, semantic_nc, None, resolution, kernel_size=1, activation="linear", up=up, conv_clamp=conv_clamp)
                self.fpn_up.append(fpn_up)
                fpn_conv = Conv2dLayer(in_ch, semantic_nc, None, resolution, kernel_size=1, activation="linear", conv_clamp=conv_clamp)
                self.fpn_out.append(fpn_conv)

            if semantic_input_mode == "progressive_input":
                self.layers.add_module(f"sematic_input{'x'.join([str(_) for _ in resolution])}", layers.SemanticCat())
                in_ch += semantic_nc
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
        if self.semantic_input_mode == "at_input":
            to_cat.append(semantic_mask)
        if self._input_condition:
            to_cat.extend([condition, mask,])
        if self.input_cse:
            to_cat.extend([embedding])
        x = torch.cat(to_cat, dim=1)
        batch = {"x": x, "semantic_mask": semantic_mask, "mask": None}
        batch = self.from_rgb(batch)
        fpn_skips = [self.fpn_out[0](batch["x"])]

        for i, layer in enumerate(self.layers):
            batch = layer(batch)
            if i < len(self.layers)-2:
                fpn_skips.append(
                    self.fpn_out[i+1](batch["x"], gain=np.sqrt(.5))
                )
            elif i == len(self.layers) - 2:
                fpn_skips.append(batch["x"])

        fpn_skips.reverse()
        segmentation = fpn_skips[0]
        for i in range(len(self.fpn_up)):
            segmentation = self.fpn_up[-i-1](segmentation, gain=np.sqrt(0.5))
            segmentation = (segmentation + fpn_skips[i+1])
        batch = self.output_layer(batch)
        segmentation = self.output_seg_layer(segmentation)
        x = batch["x"]
        out = dict(score=x, segmentation=segmentation, E=segmentation)
        if self.pred_only_cse:
            del out["segmentation"]
        if self.pred_only_semantic:
            del out["E"]
        return out

    def get_chsize(self, imsize):
        n = int(np.log2(self._max_imsize) - np.log2(imsize))
        mul = min(2 ** n, self._max_cnum_mul)
        ch = self._cnum * mul
        return int(ch)
