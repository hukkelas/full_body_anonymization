from fba.layers.base import Sequential
import numpy as np
import torch
from ..build import GENERATOR_REGISTRY
from ..layers.stylegan2_layers import Conv2dLayer
from .unet_generator import get_chsize, UnetGenerator
from . import gblocks


@GENERATOR_REGISTRY.register_module
class ComodGenerator(UnetGenerator):

    def __init__(self,
                *args,
                min_comod_res=8,
                **kwargs
                ) -> None:
        super().__init__(*args, **kwargs)
        assert not self.modulate_encoder
        assert kwargs["style_cfg"].type == "CoModStyleMapper"
        n_levels = int(np.log2(max(kwargs["imsize"])) - np.log2(kwargs["min_fmap_resolution"]))+1
        imsize = kwargs["imsize"]
        end_enc_res = [imsize[0]/2**(n_levels-1), imsize[1]/2**(n_levels-1)]
        convs = []
        ch = get_chsize(max(end_enc_res), kwargs["cnum"], max(imsize), kwargs["max_cnum_mul"])
        res = end_enc_res
        while True:
            if any([_<=min_comod_res for _ in res]):
                break
            convs.append(
                Conv2dLayer(ch, ch, res, down=2, conv_clamp=kwargs["conv_clamp"])
            )
            res = [res[0]//2, res[1]//2]
        self.comod_convs = Sequential(*convs)

    def forward(self, condition, mask, z=None, w=None, semantic_mask=None, embedding=None, border=None, vertices=None, update_ema=False, **kwargs):
        if z is None:
            z = self.get_z(condition)

        if self._input_semantic:
            x = torch.cat((condition, mask, semantic_mask), dim=1)
        elif self.input_cse:
            x = torch.cat((condition, mask, embedding), dim=1)
        else:
            x = torch.cat((condition, mask), dim=1)
        x = self.from_rgb(x)

        batch = {"x": x, "mask": mask, }

        unet_features = []
        for i, layer in enumerate(self.encoder):
            batch = layer(batch)
            if i != len(self.encoder)-1:
                unet_features.append(batch["x"])

        if hasattr(self, "middle_blocks"):
            batch = self.middle_blocks(batch)
        y = self.comod_convs(batch["x"])

        modulation_params = iter(self.style_net(comod_input=y, z=z, w=w))
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
