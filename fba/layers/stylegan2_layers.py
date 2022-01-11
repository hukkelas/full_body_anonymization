# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from typing import List
import numpy as np
import torch
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma

#----------------------------------------------------------------------------

@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

#----------------------------------------------------------------------------

@misc.profiled_function
def modulated_conv2d(
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                     # Modulation coefficients of shape [batch_size, in_channels].
    noise           = None,     # Optional noise tensor to add to the output activations.
    up              = 1,        # Integer upsampling factor.
    down            = 1,        # Integer downsampling factor.
    padding         = 0,        # Padding with respect to the upsampled image.
    resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate      = True,     # Apply weight demodulation?
    flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings(): # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(x=x, w=w, f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x

#----------------------------------------------------------------------------

class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.repr = dict(
            in_features=in_features, out_features=out_features, bias=bias,
            activation=activation, lr_multiplier=lr_multiplier, bias_init=bias_init)
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        misc.assert_shape(x, [None, self.in_features])
        w = self.weight * self.weight_gain
        b = self.bias
        if b is not None:
            b =  b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            b = b.to(x.dtype) if b is not None else None
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={item}" for key, item in self.repr.items()])


class Conv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        w_dim,                          # Intermediate latent (W) dimensionality.
        resolution: List[int],                     # Resolution of this layer.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        down = 1,                       # Integer downsampling factor
        use_noise       = False,         # Enable noise input?
        use_w = False,                   # Enable style input?
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        bias = True,
        use_norm=False,
        lr_multiplier=1,
    ):
        super().__init__()
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = torch.nn.InstanceNorm2d(out_channels)
            self.use_norm = self.norm is not None
        self.repr = dict(
            in_channels=in_channels, out_channels=out_channels,w_dim=w_dim,resolution=resolution,
            kernel_size=kernel_size, up=up, down=down, use_noise=use_noise, use_w=use_w,
            activation=activation, resample_filter=resample_filter, conv_clamp=conv_clamp, bias=bias,
            use_norm=self.use_norm
            )
        assert not use_w 
#        assert not use_noise

        assert not(down != 1 and use_w), "This was not originally implemented. Go over."
#        assert not(down != 1 and use_noise)

        self.resolution = resolution
        self.up = up
        self.down = down
        self.use_noise = use_noise
        self.use_w = use_w
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        if self.use_w:
            self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        else:
            self.weight_gain = lr_multiplier / np.sqrt(in_channels * (kernel_size ** 2))
        memory_format = torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        if use_noise:
            self.register_buffer('noise_const', torch.randn(resolution))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels])) if bias else None
        self.bias_gain = lr_multiplier

    def forward(self, x, w=None, mask=None, gamma=None, beta=None, noise_mode='random', fused_modconv=True, gain=1, **kwargs):
        assert noise_mode in ['random', 'const', 'none']
        if self.use_w:
            in_resolution = [_//self.up*self.down for _ in self.resolution]
            misc.assert_shape(x, [None, self.weight.shape[1], *in_resolution])
            styles = self.affine(w)
        else:
            misc.assert_shape(x, [None, self.weight.shape[1], None, None])
        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, *self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1) # slightly faster
        if gamma is not None:
            assert gamma.dtype == torch.float32
            x = x * gamma
        if beta is not None:
            x = x + beta

        if self.use_w:
            x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up, down=self.down,
                padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)
        else:
            w = self.weight * self.weight_gain # Equalized learning rate. Implemented in modulated conv2d
            x = conv2d_resample.conv2d_resample(
                x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down,
                padding=self.padding, flip_weight=flip_weight)
        if self.use_norm:
            x = self.norm(x)
        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        b = self.bias.to(x.dtype)*self.bias_gain if self.bias is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x


    def extra_repr(self) -> str:
        return ", ".join([f"{key}={item}" for key, item in self.repr.items()])
#----------------------------------------------------------------------------

class ToRGBLayer(torch.nn.Module):
    def __init__(self, 
        in_channels, out_channels, w_dim,
        kernel_size=1, conv_clamp=None,
        use_w=True):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.use_w = use_w
        if self.use_w:
            self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, batch, fused_modconv=True):
        batch = dict(batch) # Stop overwrite if skip connections
        x = batch["x"]
        if self.use_w:
            w = batch["w"]
            styles = self.affine(w) * self.weight_gain
            x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        else:
            w = self.weight_gain * self.weight
            x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype))
        b = self.bias.to(x.dtype) if self.bias is not None else None
        x = bias_act.bias_act(x, b, clamp=self.conv_clamp)
        batch["x"] = x
        return batch

#----------------------------------------------------------------------------

class StyleGAN2Block(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        out_channels,                       # Number of output channels.
        resolution: List[int],                         # Resolution of this block.
        w_dim=None,                              # Intermediate latent (W) dimensionality.
        architecture        = 'resnet',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        up = 1,
        down = 1,
        use_w = False,
        use_norm=False,
        **layer_kwargs,                     # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.use_w = use_w
        self.resolution = resolution
        self.architecture = architecture
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0
        self.down = down

        self.conv0 = Conv2dLayer(
            in_channels, out_channels, w_dim=w_dim, resolution=resolution,
            use_w=use_w,
            down=down,
            resample_filter=resample_filter, conv_clamp=conv_clamp,
            use_norm=use_norm, **layer_kwargs)
        self.num_conv += 1

        self.conv1 = Conv2dLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
            up = up,  use_w=use_w,
            use_norm=use_norm,
            conv_clamp=conv_clamp, **layer_kwargs)
        self.num_conv += 1


        if architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels,  w_dim, resolution, kernel_size=1, bias=False, up=up, down=down,
                resample_filter=resample_filter, use_w=use_w, **layer_kwargs)

    def forward(self, batch, fused_modconv=None, **layer_kwargs):
        batch = dict(batch) # Stop overwrite if skip connections
        if "modulation_params" in batch:
            modulation_iter = batch["modulation_params"]
        else:
            modulation_iter = iter([{}, {}])
        if self.use_w: 
            ws = batch["ws"]
            misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
            w_iter = iter(ws.unbind(dim=1))
        else:
            w_iter = iter([None]*self.num_conv)
        memory_format = torch.contiguous_format
        # Input.
        misc.assert_shape(batch["x"], [None, self.in_channels, *[_*self.down for _ in self.resolution]])
        batch["x"] = batch["x"].to(memory_format=memory_format)
        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(batch["x"], gain=np.sqrt(0.5))
            batch["x"] = self.conv0(
                x = batch["x"], w=next(w_iter), mask=batch["mask"],
                fused_modconv=fused_modconv, **layer_kwargs,
                **next(modulation_iter))
            batch["x"] = self.conv1(x = batch["x"], w=next(w_iter), mask=batch["mask"],
                fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs,
                **next(modulation_iter))
            batch["x"] = y + batch["x"]
        else:
            batch["x"] = self.conv0(
                batch["x"], next(w_iter), mask=batch["mask"], fused_modconv=fused_modconv, **layer_kwargs,
                **next(modulation_iter))
            batch["x"] = self.conv1(batch["x"], next(w_iter), mask=batch["mask"],
                fused_modconv=fused_modconv, **layer_kwargs,
                **next(modulation_iter))
        return batch


@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings(): # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        return x

#----------------------------------------------------------------------------

class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        resolution: List[int],                     # Resolution of this block.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.architecture = architecture

        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(
            in_channels + mbstd_num_channels, in_channels,
            w_dim=None, resolution=resolution,
            kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * resolution[0]*resolution[1], in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, 1)

    def forward(self, batch):
        batch = dict(batch) # Stop overwrite if skip connections
        x = batch["x"]
        misc.assert_shape(x, [None, self.in_channels, *self.resolution]) # [NCHW]
        memory_format = torch.contiguous_format

        x = x.to(memory_format=memory_format)
        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)
        batch["x"] = x
        return batch
