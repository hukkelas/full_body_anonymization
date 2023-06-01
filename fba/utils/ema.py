import torch
import copy
from fba import logger
from .torch_utils import set_requires_grad, to_cuda


class EMA:
    """
    Expoenential moving average.
    See:
        Yazici, Y. et al.The unusual effectiveness of averaging in GAN training. ICLR 2019

    """

    def __init__(
            self,
            generator: torch.nn.Module,
            batch_size: int,
            nimg_half_time: int, # Half-life of the moving average
            rampup_nimg: int # Linear scaling of moving average. Reduces startup bias
            ):
        self._rampup_nimg = rampup_nimg
        self._nimg_half_time = nimg_half_time
        self._batch_size = batch_size
        with torch.no_grad():
            self.generator = copy.deepcopy(generator.cpu()).eval()
            self.generator = to_cuda(self.generator)
            to_cuda(generator)
        self.old_ra_beta = 0
        set_requires_grad(self.generator, False)

    def update_beta(self):
        y = self._nimg_half_time
        global_step = logger.global_step
        if self._rampup_nimg != 0:
            y = min(self._nimg_half_time, self._nimg_half_time*global_step/self._rampup_nimg)
        self.ra_beta = 0.5 ** (self._batch_size/max(y, 1e-8))
        if self.ra_beta != self.old_ra_beta:
            logger.log_variable("stats/EMA_beta", self.ra_beta)
        self.old_ra_beta = self.ra_beta

    @torch.no_grad()
    def update(self, normal_G):
        with torch.autograd.profiler.record_function("EMA_update"):
            for ema_p, p in zip(self.generator.parameters(),
                                normal_G.parameters()):
                ema_p.copy_(p.lerp(ema_p, self.ra_beta))
            for ema_buf, buff in zip(self.generator.buffers(),
                                     normal_G.buffers()):
                ema_buf.copy_(buff)

    def __call__(self, *args, **kwargs):
        return self.generator(*args, **kwargs)

    def cuda(self, *args, **kwargs):
        self.generator = self.generator.cuda()
        return self

    def state_dict(self, *args, **kwargs):
        return self.generator.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.generator.load_state_dict(*args, **kwargs)

    def eval(self):
        self.generator.eval()

    def train(self):
        self.generator.train()

    @property
    def module(self):
        return self.generator.module
    
    @property
    def _z_channels(self):
        return self.generator._z_channels
