from .style_net import build_stylenet
import numpy as np
import torch
from .latent_projectors import ChannelWiseAddNoise
from . import gblocks
from .base import BaseGenerator
from ..build import GENERATOR_REGISTRY
from ..layers.stylegan2_layers import Conv2dLayer, StyleGAN2Block, ToRGBLayer
from typing import List, Optional
from ..layers import Sequential


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
                z_channels: int,
                mask_output: bool,
                input_semantic: Optional[str],
                semantic_nc: Optional[int],
                conv_clamp: int,
                style_cfg: dict,
                embed_z: bool,
                class_specific_z: bool,
                input_cse: bool,
                cse_nc: int,
                n_middle_blocks: int,
                use_cse: bool,
                latent_space: Optional[str],
                modulate_encoder: bool,
                norm_type: str,
                norm_unet: str,
                unet_skip: str,
                *args,
                **kwargs
                ) -> None:
        super().__init__(z_channels)

        if use_cse:
            semantic_nc = None
        else:
            cse_nc = None
        assert semantic_nc is None or cse_nc is None
        self.latent_space = latent_space
        semantic_nc = 0 if semantic_nc is None else semantic_nc
        cse_nc = 0 if cse_nc is None else cse_nc
        semantic_nc += cse_nc
        self.imsize = imsize
        self.class_specific_z = class_specific_z
        self._cnum = cnum
        self._max_cnum_mul = max_cnum_mul
        self._min_fmap_resolution = min_fmap_resolution
        self._image_channels = image_channels
        self._max_imsize = max(imsize)
        self._input_semantic = input_semantic
        self._z_channels = z_channels
        self.semantic_nc = semantic_nc
        self.input_cse = input_cse
        self.embed_z = embed_z
        self.modulate_encoder = modulate_encoder
        self.norm_unet = norm_unet
        self.norm_type = norm_type
        self._encoder_out_shape = [
            get_chsize(min_fmap_resolution, cnum, self._max_imsize, max_cnum_mul),
            min_fmap_resolution, min_fmap_resolution]
        n_levels = int(np.log2(self._max_imsize) - np.log2(min_fmap_resolution))+1
        encoder_layers = []
        if self.embed_z:
            self.z_projector = ChannelWiseAddNoise(
                z_channels, fmap_size=[s//2**(n_levels-1) for s in imsize])
        self.from_rgb = Conv2dLayer(
            image_channels + 1 + semantic_nc*input_semantic + input_cse*cse_nc,
            cnum, self.imsize, 1
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
                down=down, conv_clamp=conv_clamp,
                norm_type=self.norm_type)
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

            up = 1
            if i != n_levels - 1:
                up = 2
            block = StyleGAN2Block(
                in_ch, out_ch, resolution, up=up, conv_clamp=conv_clamp,
                norm_type=self.norm_type)
            decoder_layers.append(block)
            if i != 0:
                if unet_skip == "residual":
                    unet_block = Conv2dLayer(
                        in_ch, in_ch, resolution, kernel_size=1, conv_clamp=conv_clamp,
                        norm_type=self.norm_unet)
                else:
                    unet_block = Conv2dLayer(
                        in_ch*2, in_ch, resolution, kernel_size=1, conv_clamp=conv_clamp,
                        norm_type=self.norm_unet)
                setattr(self, f"unet_block{i}", unet_block)

            feature_sizes_dec.append([in_ch, *resolution])
            feature_sizes_dec.append([out_ch, *resolution])
        self.unet_skip = unet_skip
        # Initialize "middle blocks" that do not have down/up sample
        middle_blocks = []
        feature_sizes_mid = []
        for i in range(n_middle_blocks):
            resolution = [x//2**(n_levels-1) for x in imsize]
            ch = get_chsize(min_fmap_resolution, cnum, self._max_imsize, max_cnum_mul)
            block = StyleGAN2Block(
                ch, ch, resolution, conv_clamp=conv_clamp,
                norm_type=norm_type)
            middle_blocks.append(block)
            feature_sizes_mid.append([ch, *resolution])
            feature_sizes_mid.append([ch, *resolution])
        if n_middle_blocks != 0:
            self.middle_blocks = Sequential(*middle_blocks)
        self.decoder = torch.nn.ModuleList(decoder_layers)
        self.to_rgb = ToRGBLayer(cnum, image_channels, conv_clamp=conv_clamp)

        # Initialize "middle blocks" that do not have down/up sample
        self.decoder = torch.nn.ModuleList(decoder_layers)

        self.scale_grad = scale_grad
        self.mask_output = mask_output

        self.style_net = build_stylenet(style_cfg,
            feature_sizes_enc=[],
            feature_sizes_dec=feature_sizes_dec,
            feature_sizes_mid=[],
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

        modulation_params = iter(self.style_net(
            semantic_mask=semantic_mask, z=z, mask=mask, border=border,
            embedding=embedding, vertices=vertices, w=w, update_ema=update_ema))
        if self._input_semantic:
            x = torch.cat((condition, mask, semantic_mask), dim=1)
        elif self.input_cse:
            x = torch.cat((condition, mask, embedding), dim=1)
        else:
            x = torch.cat((condition, mask), dim=1)
        x = self.from_rgb(x)

        batch = {"x": x, "mask": mask, }
        if self.modulate_encoder:
            batch["modulation_params"] = modulation_params
        unet_features = []
        for i, layer in enumerate(self.encoder):
            batch = layer(batch)
            if i != len(self.encoder)-1:
                unet_features.append(batch["x"])
        if hasattr(self, "middle_blocks"):
            batch = self.middle_blocks(batch)
        if not self.modulate_encoder:
            batch["modulation_params"] = modulation_params

        if self.embed_z:
            batch = self.z_projector(batch, z)
        for i, layer in enumerate(self.decoder):
            if i != 0:
                unet_layer = getattr(self, f"unet_block{i}")
                if self.unet_skip == "residual":
                    batch["x"] = batch["x"] + unet_layer(unet_features[-i], gain=np.sqrt(.5))
                else:
                    x_skip = torch.cat((batch["x"], unet_features[-i]), dim=1)
                    batch["x"] = unet_layer(x_skip)
            batch = layer(batch)
        x = self.to_rgb(batch)["x"]
        if self.mask_output:
            x = gblocks.mask_output(True, condition, x, mask)
        return dict(img=x)

    def get_w(self, z):
        if self.latent_space == "w_cse":
            return self.style_net.E_map.map_network(z)
        if self.latent_space == "w_stylegan2":
            return self.style_net.w_map(z)
        if self.latent_space == "w_comodgan":
            return self.style_net.w_map(z)
        raise ValueError()


