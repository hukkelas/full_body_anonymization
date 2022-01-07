import torch
from ..layers import Module
from fba import utils

class BaseGenerator(Module):

    def __init__(self, z_channels: int):
        super().__init__()
        self.z_channels = z_channels

    @torch.no_grad()
    def get_z(
            self,
            x: torch.Tensor = None,
            z: torch.Tensor = None,
            truncation_value: float = None,
            batch_size: int = None,
            dtype=None, device=None) -> torch.Tensor:
        """Generates a latent variable for generator. 
        """
        if z is not None:
            return z
        if x is not None:
            batch_size = x.shape[0]
            dtype = x.dtype
            device = x.device
        if device is None:
            device = utils.get_device()
        if truncation_value == 0:
            return torch.zeros((batch_size, self.z_channels), device=device, dtype=dtype)
        z = torch.randn((batch_size, self.z_channels), device=device, dtype=dtype)
        if truncation_value is None:
            return z
        while z.abs().max() > truncation_value:
            m = z.abs() > truncation_value
            z[m] = torch.rand_like(z)[m]
        return z
