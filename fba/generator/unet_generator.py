from .style_net import build_stylenet
import numpy as np
import torch
from .latent_projectors import ChannelWiseAddNoise
from . import gblocks
from .base import BaseGenerator
from .. import layers
from ..build import GENERATOR_REGISTRY
from ..layers.stylegan2_layers import Conv2dLayer, StyleGAN2Block, ToRGBLayer
from typing import List, Optional


def get_chsize(imsize, cnum, max_imsize, max_cnum_mul):
    n = int(np.log2(max_imsize) - np.log2(imsize))
    mul = min(2**n, max_cnum_mul)
    ch = cnum * mul
    return int(ch)



@GENERATOR_REGISTRY.register_module
class UnetGenerator(BaseGenerator):

    def __init__(
                # See configs/segan/base.py for default values
                self,
                scale_grad: bool,
                image_channels: int,
                min_fmap_resolution: int, 
                imsize: List[int],
                cnum: int,
                max_cnum_mul: int,
                n_middle_blocks: int, # Number of BasicBlocks to incldue at minimum resolution
                z_channels: int,
                mask_output: bool,
                semantic_input_mode: Optional[str],
                semantic_nc: Optional[int],
                conv_clamp: int,
                use_norm: bool,
                style_cfg: dict,
                embed_z: bool,
                class_specific_z: bool,
                input_cse: bool,
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
        self._min_fmap_resolution = min_fmap_resolution
        self._image_channels = image_channels
        self._max_imsize = max(imsize)
        self._semantic_input_mode = semantic_input_mode
        self._z_channels = z_channels
        self.semantic_nc = semantic_nc
        self.input_cse = input_cse
        self.embed_z = embed_z
        self._encoder_out_shape = [
            get_chsize(min_fmap_resolution, cnum, self._max_imsize, max_cnum_mul),
            min_fmap_resolution, min_fmap_resolution]
        n_levels = int(np.log2(self._max_imsize) - np.log2(min_fmap_resolution))+1
        encoder_layers = []
        if self.embed_z:
            self.z_projector = ChannelWiseAddNoise(
                z_channels, fmap_size=[s//2**(n_levels-1) for s in imsize])
        self.from_rgb = Conv2dLayer(
            image_channels + 1 + semantic_nc*(semantic_input_mode == "at_input") + input_cse*cse_nc,
            cnum, w_dim, self.imsize, 1, use_w=use_w, use_noise=use_noise
        )
        feature_sizes_enc = []
        for i in range(n_levels): # Encoder layers
            resolution = [x//2**i for x in imsize]
            in_ch = get_chsize(max(resolution), cnum, self._max_imsize, max_cnum_mul)
            second_ch = in_ch
            out_ch = get_chsize(max(resolution)//2, cnum, self._max_imsize, max_cnum_mul)
            down = 2

            if i == 0: # first (lowest) block. Downsampling is performed at the start of the block
                down = 1
            if i == n_levels - 1:
                out_ch = second_ch
            feature_sizes_enc.append([in_ch, *[_*down for _ in resolution]]) # Used for modulation before convolution
            feature_sizes_enc.append([out_ch, *[_ for _ in resolution]])
            block = StyleGAN2Block(
                in_ch, out_ch, resolution=resolution,
                down=down, use_w=use_w, conv_clamp=conv_clamp, use_noise=use_noise,
                use_norm=use_norm)
            encoder_layers.append(block)
        self.encoder = torch.nn.ModuleList(encoder_layers)

        # initialize decoder
        decoder_layers = []
        feature_sizes_dec = []
        for i in range(n_levels):
            resolution = [x//2**(n_levels-1-i) for x in imsize]
            in_ch = get_chsize(max(resolution)//2, cnum, self._max_imsize, max_cnum_mul)
            out_ch = get_chsize(max(resolution), cnum, self._max_imsize, max_cnum_mul)
            if i == 0:  # first (lowest) block
                in_ch = get_chsize(max(resolution), cnum, self._max_imsize, max_cnum_mul)
            if semantic_input_mode == "progressive_input":
                in_ch += semantic_nc
                decoder_layers.append(layers.SemanticCat())

            up = 1
            if i != n_levels - 1:
                up = 2
            block = StyleGAN2Block(
                in_ch, out_ch, resolution, w_dim, up=up, use_w=use_w, conv_clamp=conv_clamp, use_noise=use_noise,
                use_norm=use_norm)
            decoder_layers.append(block)
            if i != 0:
                unet_block = Conv2dLayer(
                    in_ch*2, in_ch, w_dim, resolution, kernel_size=1, conv_clamp=conv_clamp, use_noise=use_noise,
                    use_norm=use_norm)
                setattr(self, f"unet_block{i}", unet_block)
                feature_sizes_dec.append([in_ch*2, *resolution]) # unet layer

            feature_sizes_dec.append([in_ch, *resolution])
            feature_sizes_dec.append([out_ch, *resolution])

        self.to_rgb = ToRGBLayer(cnum, image_channels, w_dim, conv_clamp=conv_clamp, use_w=use_w)

        # Initialize "middle blocks" that do not have down/up sample
        feature_sizes_mid = []
        self.middle_blocks = []
        for i in range(n_middle_blocks):
            resolution = [x//2**(n_levels-1) for x in imsize]
            ch = get_chsize(min_fmap_resolution, cnum, self._max_imsize, max_cnum_mul)
            feature_sizes_mid.append([ch, *resolution])
            feature_sizes_mid.append([ch, *resolution])
            block = StyleGAN2Block(
                ch, ch, resolution, w_dim, conv_clamp=conv_clamp, use_w=use_w, use_noise=use_noise,
                use_norm=use_norm)
            self.middle_blocks.append(block)
        self.middle_blocks = layers.Sequential(*self.middle_blocks)
        self.decoder = torch.nn.ModuleList(decoder_layers)

        self.scale_grad = scale_grad
        self.mask_output = mask_output

        self.style_net = build_stylenet(style_cfg,
            feature_sizes_enc=feature_sizes_enc,
            feature_sizes_dec=feature_sizes_dec,
            feature_sizes_mid=feature_sizes_mid,
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

    def forward(self, condition, mask, z=None, w=None, semantic_mask=None, embedding=None, border=None, vertices=None, update_ema=False, **kwargs):
        if z is None:
            z = self.get_z(condition)
        if self._semantic_input_mode == "at_input":
            x = torch.cat((condition, mask, semantic_mask), dim=1)
        elif self.input_cse:
            x = torch.cat((condition, mask, embedding), dim=1)
        else:
            x = torch.cat((condition, mask), dim=1)
        x = self.from_rgb(x)
        modulation_params = self.style_net(
            semantic_mask=semantic_mask, z=z, mask=mask, border=border,
            embedding=embedding, vertices=vertices, w=w, update_ema=update_ema)
        batch = {"x": x, "mask": mask, "modulation_params": iter(modulation_params)}
        unet_features = []
        for i, layer in enumerate(self.encoder):
            batch = layer(batch)
            if i != len(self.encoder)-1:
                unet_features.append(batch["x"])
        outs = {}
        if self.embed_z:
            batch = self.z_projector(batch, z)
        batch = self.middle_blocks(batch)
        for i, layer in enumerate(self.decoder):
            if i != 0:
                x_skip = torch.cat((batch["x"], unet_features[-i]), dim=1)
                unet_layer = getattr(self, f"unet_block{i}")
                batch["x"] = unet_layer(x_skip, **next(batch["modulation_params"]))
            batch = layer(batch)
        x = self.to_rgb(batch)["x"]
        if self.mask_output:
            x = gblocks.mask_output(True, condition, x, mask)
        outs["img"] = x
        return outs
