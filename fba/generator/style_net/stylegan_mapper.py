import torch
import tqdm
from fba.layers import Module
from fba.layers.stylegan2_layers import (FullyConnectedLayer, normalize_2nd_moment)
from torch import nn
from .build import STYLE_ENCODER_REGISTRY


class LatentMapper(Module):
    def __init__(
            self,
            z_channels,
            out_dim=512,
            n_layer=2,
            ):
        super().__init__()
        self.out_dim = out_dim
        self.z_channels = z_channels
        self.n_layer = n_layer

        for i in range(n_layer):
            in_ch = z_channels if i == 0 else out_dim
            layer = FullyConnectedLayer(in_ch, out_dim, activation="lrelu")
            setattr(self, f"fromZ{i}", layer)

    def forward(self, z):
        z = normalize_2nd_moment(z)
        for i in range(self.n_layer):
            layer = getattr(self, f"fromZ{i}")
            z = layer(z)
        return z

    def device(self):
        return self.fromE0.weight.device

    @torch.no_grad()
    def update_average_w(self, n):
        assert n > 0
        avg_w = self.forward(torch.randn(1, self.z_channels, device=self.device()))
        for i in tqdm.trange(n-1):
            w = self.forward(torch.randn(1, self.z_channels, device=self.device()))
            avg_w.copy_(w.lerp(avg_w, 0.995))
        if hasattr(self, "w_avg"):
            self.w_avg.copy_(avg_w)
        else:
            self.register_buffer("w_avg", avg_w)

    @torch.no_grad()
    def get_average_w(self, n=10000):
        if not hasattr(self, "w_avg"):
            self.update_average_w(n)
        return self.w_avg


@STYLE_ENCODER_REGISTRY.register_module
class StyleGANMapper(Module):

    def __init__(
            self,
            feature_sizes_enc,
            feature_sizes_dec,
            feature_sizes_mid,
            z_channels: int,
            only_gamma,
            included_features=None,
             **kwargs
            ):
        super().__init__()
        self.feature_sizes_dec =list(feature_sizes_mid) +  list(feature_sizes_dec) 
        self.w_map = LatentMapper(z_channels=z_channels)
        self.only_gamma = only_gamma
        self.dec_layers = nn.ModuleList(
            [FullyConnectedLayer(self.w_map.out_dim, shape[0]*(1+ int(not only_gamma)), activation="linear", bias_init=int(only_gamma))
             for idx, shape in enumerate(self.feature_sizes_dec)])
        self.included_features = included_features
        self.min_fmap_H = min([_[1] for _ in feature_sizes_enc+feature_sizes_dec+feature_sizes_mid])

    def forward(self, z, w=None, **kwargs):
        if w is None:
            w = self.w_map(z)
        mod_params = []
        for layer in self.dec_layers:
            gamma = layer(w).view(w.shape[0], -1, 1, 1)
            if self.only_gamma:
                mod_params.append({"gamma": gamma})
                continue
            gamma, beta = gamma.chunk(2, dim=1)
            gamma = gamma + 1
            mod_params.append({"gamma": gamma, "beta": beta})
        return mod_params
