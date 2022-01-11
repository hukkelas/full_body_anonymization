import torch.nn.functional as F
import tqdm
import torch
from fba.layers import Module
from fba.utils.utils import get_embed_stats
from fba.layers.stylegan2_layers import Conv2dLayer, FullyConnectedLayer, normalize_2nd_moment
from torch import nn
from fba import logger
from .build import STYLE_ENCODER_REGISTRY
from torch_utils.misc import assert_shape


class CSELatentMapper(Module):
    def __init__(
            self,
            E_dim,                      # Input embedding (E) dimensionality, 0 = no embedding.
            hidden_dim=512,
            out_dim=512,
            n_layer_combined=6,
            n_layer_z=1,
            n_layer_e=1,
            activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
            lr_multiplier=1,     # Learning rate multiplier for the mapping layers.
            lr_z=1,
            input_z=False,
            z_channels=None,
            normalize_z=True,
            n_vertices=27554,
            residual=True,
            ema_beta=0.995,
            has_trained_ema=True
            ):
        super().__init__()
        self.E_dim = E_dim
        self.n_layer_combined = n_layer_combined
        self.n_layer_z = n_layer_z
        self.n_layer_e = n_layer_e
        self.out_dim = out_dim
        self.input_z = input_z
        self.normalize_z = normalize_z
        self.n_vertices = n_vertices
        self.z_channels = z_channels
        self.hidden_dim = hidden_dim
        self.residual = residual
        self.w_avg_beta = ema_beta
        E_mean, E_rstd, E = get_embed_stats()
        if E_mean is None:
            logger.warn("Did not find E_mean/E_rstd in cache. If you're NOT running a training script, this is fine (will be overwritten by checkpoint load).")
            self.register_buffer("E", torch.zeros(n_vertices, E_dim))
        else:
            E = (E-E_mean)*E_rstd
            self.register_buffer("E", E)
        assert self.E.shape == (n_vertices, E_dim), ((n_vertices, E_dim), self.E.shape)
        for i in range(n_layer_e):
            layer = FullyConnectedLayer(E_dim, hidden_dim, lr_multiplier=lr_multiplier, activation=activation)
            E_dim = hidden_dim
            setattr(self, f"fromE{i}", layer)        
        if input_z:
            for i in range(n_layer_z):
                layer = FullyConnectedLayer(z_channels, z_channels, activation=activation, lr_multiplier=lr_z)
                setattr(self, f"fromZ{i}", layer)
        features_list = [input_z*z_channels + hidden_dim] + [hidden_dim] * (n_layer_combined - 1) + [out_dim]
        for idx in range(n_layer_combined):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, lr_multiplier=lr_multiplier, activation=activation)
            if self.residual and (idx+1) % 2 == 0:
                res_layer = FullyConnectedLayer(features_list[idx-1], out_features, lr_multiplier=lr_multiplier, activation="linear")
                setattr(self, f"residual{idx}", res_layer)
            setattr(self, f'fc{idx}', layer)
        if self.n_layer_combined == 0:
            self.out_dim = hidden_dim + z_channels*(self.input_z)

        if has_trained_ema:
            self.register_buffer("w_avg", torch.zeros((n_vertices, self.out_dim)))

    def map_network(self, z):
        # Embed, normalize, and concat inputs.
        E = self.E
        for i in range(self.n_layer_e):
            layer = getattr(self, f"fromE{i}")
            E = layer(E)
        if self.input_z:
            if self.normalize_z:
                z = normalize_2nd_moment(z)
            for i in range(self.n_layer_z):
                layer = getattr(self, f"fromZ{i}")
                z = layer(z)
            z = z[:, None].repeat(1, self.n_vertices, 1)
            E = E[None].repeat(z.shape[0], 1, 1)
            # Ravel the embedding from NxKxC -> N*KxC
            x = torch.cat((E, z), dim=2).view(self.n_vertices*E.shape[0], -1)
        else:
            x = E
        # Main layers.
        residual = x
        for idx in range(self.n_layer_combined):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)
            if self.residual and (idx+1) % 2 == 0:
                layer = getattr(self, f"residual{idx}")
                residual = layer(residual)
                x = (x + residual)/(2**0.5)
                residual = x
        return x

    def forward(self, vertices, z, w, update_ema):
        if w is None:
            w = self.map_network(z)
        else:
            assert not self.training
        if update_ema and self.input_z:
            if self.input_z:
                self.w_avg.copy_(w.float().detach().mean(dim=0).lerp(self.w_avg.float(), self.w_avg_beta))
        indices = vertices.view(-1, *vertices.shape[-2:])
        assert_shape(indices, [None, None, None])
        if self.input_z:
            # Torch memory is column major.
            #For inputted z, E is raveled to NKxC, thus indices are offset by K*batch_idx
            indices = indices + torch.arange(0, vertices.shape[0], device=vertices.device)[:, None, None]*self.n_vertices
        E = F.embedding(indices.long(), w).permute(0, 3, 1, 2)
        return E

    def device(self):
        return self.fromE0.weight.device

    @torch.no_grad()
    def update_average_w(self, n):
        assert n > 0
        avg_w = self.map_network(torch.randn(1, self.z_channels, device=self.device()))
        for i in tqdm.trange(n-1):
            w = self.map_network(torch.randn(1, self.z_channels, device=self.device()))
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
class CSEStyleMapper(Module):

    def __init__(
            self,
            feature_sizes_enc,
            feature_sizes_dec,
            feature_sizes_mid,
            w_mapper: dict,
            cse_nc: int,
            z_channels: int,
            modulate_middle=True,
            modulate_encoder=True,
             **kwargs
            ):
        super().__init__()
        self.modulate_middle = modulate_middle
        self.feature_sizes_dec = list(feature_sizes_dec)
        self.feature_size_mid = feature_sizes_mid
        self.feature_size_enc = feature_sizes_enc
        self.modulate_encoder = modulate_encoder
        self.E_map = CSELatentMapper(E_dim=cse_nc, z_channels=z_channels, **w_mapper)
        self.global_mask = Conv2dLayer(
            self.E_map.out_dim + 3, self.E_map.out_dim, None, None,
            kernel_size=1, bias=False
        )
        self.global_mask.weight[:, -3:].data.fill_(0)
        self.dec_layers = [CSELinear(self.E_map.out_dim, shape[0], **kwargs) for shape in self.feature_sizes_dec]
        self.dec_layers = nn.ModuleList(self.dec_layers)
        if self.modulate_middle:
            self.mid_layers = nn.ModuleList([CSELinear(self.E_map.out_dim, shape[0], **kwargs) for shape in feature_sizes_mid])
        if self.modulate_encoder:
            self.enc_layers = nn.ModuleList([CSELinear(self.E_map.out_dim, shape[0], **kwargs) for shape in feature_sizes_enc])

    def forward(self, embedding, mask, border, z, vertices, w, update_ema, **kwargs):
        modulation_params = []
        embeddings_ = {}
        embedding = self.E_map(vertices, z, w, update_ema=update_ema)
        if embedding.shape[2] != mask.shape[2]:
            embedding = F.interpolate(embedding, size=mask.shape[2:], mode=self.resample, align_corners=True)
        E_mask = 1 - mask - border
        x = torch.cat((embedding*E_mask, mask, border, E_mask), dim=1)
        embedding = self.global_mask(x)
        if self.modulate_encoder:
            for shape, layer in zip(self.feature_size_enc, self.enc_layers):
                assert embedding.shape[2] >= shape[1], (embedding.shape, shape)
                assert embedding.shape[3] >= shape[2]
                if embedding.shape[2] != shape[1]:
                    embedding = F.interpolate(embedding, scale_factor=.5, mode="bilinear", recompute_scale_factor=False, align_corners=True)
                embeddings_[tuple(shape[1:])] = dict(embedding=embedding)
                gamma, beta = layer(**embeddings_[tuple(shape[1:])])
                modulation_params.append({"gamma": gamma, "beta": beta})
        else:
            modulation_params.extend([{}]*len(self.feature_size_enc))
        if self.modulate_middle:
            for shape, layer in zip(self.feature_size_mid, self.mid_layers):
                gamma, beta = layer(**embeddings_[tuple(shape[1:])])
                modulation_params.append({"gamma": gamma, "beta": beta})
        else:
            modulation_params.extend([{}]*len(self.feature_size_mid))
        for shape, layer in zip(self.feature_sizes_dec, self.dec_layers):
            gamma, beta = layer(**embeddings_[tuple(shape[1:])])
            modulation_params.append({"gamma": gamma, "beta": beta})
        return modulation_params


class CSELinear(nn.Module):
    def __init__(self, w_dim, num_features, **kwargs):
        super().__init__()
        self.w_dim = w_dim
        self.mlp = Conv2dLayer(
            w_dim, num_features*2, None, resolution=None, activation="linear",
            kernel_size=1, bias=True
        )

    def forward(self, embedding):
        gamma, beta = self.mlp(embedding).chunk(2, dim=1)
        return gamma.float()+1, beta


@STYLE_ENCODER_REGISTRY.register_module
class UnconditionalCSEStyleMapper(Module):

    def __init__(
            self,
            feature_sizes_dec,
            w_mapper: dict,
            cse_nc: int,
            z_channels: int,
             **kwargs
            ):
        super().__init__()
        self.feature_sizes_dec = list(feature_sizes_dec)
        self.E_map = CSELatentMapper(E_dim=cse_nc, z_channels=z_channels, **w_mapper)
        self.global_mask = Conv2dLayer(
            self.E_map.out_dim + 2, self.E_map.out_dim, None, None,
            kernel_size=1, bias=False
        )
        self.global_mask.weight[:, -2:].data.fill_(0)
        self.feature_sizes_dec.reverse()
        self.dec_layers = [CSELinear(self.E_map.out_dim, shape[0], **kwargs) for shape in self.feature_sizes_dec]
        self.dec_layers = nn.ModuleList(self.dec_layers)

    def forward(self, z, vertices,w, E_mask, update_ema,E=None, **kwargs):
        modulation_params = []
        embeddings_ = {}
        if E is None:
            embedding = self.E_map(vertices, z, w, update_ema=update_ema)
        else:
            embedding = E
        if embedding.shape[2] != E_mask.shape[2]:
            embedding = F.interpolate(embedding, size=E_mask.shape[2:], mode=self.resample, align_corners=True)
        x = torch.cat((embedding*E_mask, E_mask, 1-E_mask), dim=1)
        embedding = self.global_mask(x)
        for shape, layer in zip(self.feature_sizes_dec, self.dec_layers):
            assert embedding.shape[2] >= shape[1], (embedding.shape, shape)
            assert embedding.shape[3] >= shape[2]
            if embedding.shape[2] != shape[1]:
                embedding = F.interpolate(embedding, scale_factor=.5, mode="bilinear", recompute_scale_factor=False, align_corners=True)
            embeddings_[tuple(shape[1:])] = dict(embedding=embedding)
            gamma, beta = layer(**embeddings_[tuple(shape[1:])])
            modulation_params.append({"gamma": gamma, "beta": beta})
        modulation_params.reverse()
        return modulation_params
