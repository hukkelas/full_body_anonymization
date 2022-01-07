from fba import utils
from fba.utils import build_from_cfg, Registry

DISCRIMINATOR_REGISTRY = Registry("DISCRIMINATOR_REGISTRY")
GENERATOR_REGISTRY = Registry("GENERATOR_REGISTRY")


def build_discriminator(cfg):
    discriminator = build_from_cfg(
        cfg.discriminator,
        DISCRIMINATOR_REGISTRY,
        imsize=cfg.imsize,
        image_channels=cfg.image_channels,
        semantic_nc=cfg.semantic_nc,
        cse_nc=cfg.cse_nc
        )
    discriminator = utils.to_cuda(discriminator)
    return discriminator


def build_generator(cfg):
    generator = build_from_cfg(
        cfg.generator,
        GENERATOR_REGISTRY,
        imsize=cfg.imsize,
        image_channels=cfg.image_channels,
        semantic_nc=cfg.semantic_nc,
        cse_nc=cfg.cse_nc
        )
    generator = utils.to_cuda(generator)
    return generator
