import torch.nn.functional as F
import torch
import numpy as np
from fba.layers.stylegan2_layers import Conv2dLayer, FullyConnectedLayer
from torch import nn
from .build import STYLE_ENCODER_REGISTRY


def get_out(type: str):
    if type == "SPADE":
        return SPADEOut
    if type == "CLADE":
        return CLADEOut
    if type == "INADE":
        return ClassINADEOut
    raise ValueError(f"{type} has to be one of SPADE, CLADE, ClassINADEOut, or INADE")


@STYLE_ENCODER_REGISTRY.register_module
class SemanticStyleEncoder(nn.Module):

    def __init__(
            self,
            encoder_modulator: str,
            middle_modulator: str,
            decoder_modulator: str,
            feature_sizes_enc, feature_sizes_dec, feature_sizes_mid,
            semantic_nc, **kwargs
            ):
        super().__init__()
        self.modulate_middle = middle_modulator is not None
        self.feature_sizes_dec = list(feature_sizes_dec)
        self.feature_size_mid = feature_sizes_mid
        self.feature_size_enc = feature_sizes_enc
        self.modulate_encoder = encoder_modulator is not None
        self.dec_layers = [get_out(decoder_modulator)(semantic_nc, shape[0], **kwargs) for shape in self.feature_sizes_dec]
        self.dec_layers = nn.ModuleList(self.dec_layers)
        if self.modulate_middle:
            self.mid_layers = nn.ModuleList([get_out(middle_modulator)(semantic_nc, shape[0], **kwargs) for shape in feature_sizes_mid])
        if self.modulate_encoder:
            self.enc_layers = nn.ModuleList([get_out(encoder_modulator)(semantic_nc, shape[0], **kwargs) for shape in feature_sizes_enc])

    def forward(self, semantic_mask, z, **kwargs):
        modulation_params = []
        semantic_mask_ = semantic_mask
        if self.modulate_encoder:
            for shape, layer in zip(self.feature_size_enc, self.enc_layers):
                semantic_mask_ = F.interpolate(semantic_mask_, size=shape[1:], mode="nearest")
                gamma, beta = layer(semantic_mask_, z=z)
                modulation_params.append({"gamma": gamma, "beta": beta})
        else:
            modulation_params.extend([{}]*len(self.feature_size_enc))
        if self.modulate_middle:
            for shape, layer in zip(self.feature_size_mid, self.mid_layers):
                semantic_mask_ = F.interpolate(semantic_mask_, size=shape[1:], mode="nearest")
                gamma, beta = layer(semantic_mask_, z=z)
                modulation_params.append({"gamma": gamma, "beta": beta})
        else:
            modulation_params.extend([{}]*len(self.feature_size_mid))
        for shape, layer in zip(self.feature_sizes_dec, self.dec_layers):
            semantic_mask_ = F.interpolate(semantic_mask, size=shape[1:], mode="nearest")
            gamma, beta = layer(semantic_mask_, z=z)
            modulation_params.append({"gamma": gamma, "beta": beta})
        return modulation_params


class SPADEOut(nn.Module):

    def __init__(self, semantic_nc, num_features, nhidden, **kwargs):
        super().__init__()
        self.mlp_shared = Conv2dLayer(semantic_nc, nhidden, None, resolution=None)
        self.mlp = Conv2dLayer(
            nhidden, num_features*2, None, resolution=None, activation="linear",
            kernel_size=1
            )

    def forward(self, semantic_mask, **kwargs):
        actv = self.mlp_shared(semantic_mask)
        actv = self.mlp(actv.float())
        gamma, beta = actv.chunk(2, dim=1)

        # Fix initialization issues from original authors.
        return gamma.float()+1, beta.float()


class CLADEOut(nn.Module):
    def __init__(self, semantic_nc, num_features, **kwargs):
        super().__init__()
        self.semantic_nc = semantic_nc
        self.num_features = num_features
        self.gamma = Conv2dLayer(self.semantic_nc, self.num_features, None, None, kernel_size=1, activation="linear", bias=False)
        nn.init.constant_(self.gamma.weight, np.sqrt(self.semantic_nc))
        self.beta = Conv2dLayer(self.semantic_nc, self.num_features, None, None, kernel_size=1, activation="linear", bias=False)
        nn.init.constant_(self.beta.weight, 0.0)

    def forward(self, segmap, **kwargs):
        with torch.cuda.amp.autocast(enabled=False):
            gamma = self.gamma(segmap)
            beta = self.beta(segmap)
        return gamma.float() + 1, beta

class ClassINADEOut(nn.Module):
    def __init__(self, semantic_nc, num_features, z_channels, **kwargs):
        super().__init__()
        self.semantic_nc = semantic_nc
        self.num_features = num_features
        self.a = Conv2dLayer(self.semantic_nc, self.num_features*2, None, None, kernel_size=1, activation="linear", bias=False)
        # Init a to normal distributed since segmap is one-hot
        nn.init.normal_(self.a.weight)
        self.a.weight_gain = 1 # Gain is 1 because one-hot semantic 
        self.b = Conv2dLayer(self.semantic_nc, self.num_features*2, None, None, kernel_size=1, activation="linear", bias=False)
        nn.init.constant_(self.b.weight, 0.0)

        self.noise_affine = Conv2dLayer(
            z_channels, num_features*2, None, resolution=None, activation="linear",
            kernel_size=1
            )

    def forward(self, segmap, z, **kwargs):
        z = z.permute(0, 2, 1)
        z = torch.einsum("ncl, nlhw->nchw", z, segmap)
        z = self.noise_affine(z)
        actv = self.a(segmap)
        actv = actv * z 
        actv = actv + self.b(segmap)
        gamma, beta = actv.chunk(2, dim=1)
        return gamma.float()+1, beta
