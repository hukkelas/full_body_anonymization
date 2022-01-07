from fba.utils import Registry, build_from_cfg
from torch import nn

STYLE_ENCODER_REGISTRY = Registry("STYLE_ENCODER_REGISTRY")


def build_stylenet(style_cfg, **kwargs):
    return build_from_cfg(style_cfg, STYLE_ENCODER_REGISTRY, **kwargs)


@STYLE_ENCODER_REGISTRY.register_module
class NoneStyle(nn.Module):
    def __init__(
            self, feature_sizes_enc, feature_sizes_dec, feature_sizes_mid, **kwargs
            ):
        super().__init__()
        self.num_conv = len(feature_sizes_enc) + len(feature_sizes_dec) + len(feature_sizes_mid)

    def forward(self, *args, **kwargs):
        return iter([{}] * self.num_conv)