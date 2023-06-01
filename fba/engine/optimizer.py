import typing
import torch
from fba import logger

optimizers = {"Adam": torch.optim.Adam}



def build_optimizers(
        generator, discriminator, D_opts, G_opts,
        lazy_regularization: bool, lazy_reg_interval: int
        ) -> typing.List[torch.optim.Optimizer]:
    betas_d = D_opts.betas
    lr_d = D_opts.lr
    if lazy_regularization:
        # From Analyzing and improving the image quality of stylegan, CVPR 2020
        c = lazy_reg_interval / (lazy_reg_interval + 1)
        betas_d = [beta ** c for beta in betas_d]
        lr_d *= c
    logger.log_variable("stats/lr_D", lr_d)
    logger.log_variable("stats/lr_G", G_opts.lr)
    D_optimizer = optimizers[D_opts.type](
        discriminator.parameters(), lr=lr_d, betas=betas_d)
    G_optimizer = optimizers[G_opts.type](
        generator.parameters(),
        lr=G_opts.lr, betas=G_opts.betas)
    return G_optimizer, D_optimizer
