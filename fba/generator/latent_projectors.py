from typing import List
import torch.nn as nn
from .. import layers


class ChannelWiseAddNoise(nn.Module):

    def __init__(
            self,
            z_channels: int,
            fmap_size: List[int], # Size of feature map to concatenate Z with
            **kwargs) -> None:
        super().__init__()
        self.fmap_size = fmap_size
        self.projector = layers.FullyConnectedLayer(
            z_channels, fmap_size[0]*fmap_size[1],
            bias=False,
        )

    def forward(self, batch, z):
        z = self.projector(z)
        z = z.reshape((-1, 1, *self.fmap_size))
        if not self.training and (z.shape[2] != self.fmap_size[0] or z.shape[3] != self.fmap_size[1]):
            z = nn.functional.interpolate(z, size=batch["x"].shape[2:], mode="bilinear", align_corners=False)
        batch["x"] = batch["x"] + z
        return batch
