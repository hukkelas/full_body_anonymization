from fba.utils.torch_utils import mask2torch
import torch
import tqdm
import numpy as np
import enum
from fba import logger, utils
from fba import build_generator, build_discriminator
from fba.engine.checkpointer import load_checkpoint, get_closest_ckpt


def build_trained_generator(cfg, global_step=None):
    try:
        ckpt_path = get_closest_ckpt(cfg.checkpoint_dir, global_step)
    except FileNotFoundError as e:
        if cfg.checkpoint_url is None:
            raise e
        ckpt_path = utils.download_file(cfg.checkpoint_url)  
    ckpt = load_checkpoint(ckpt_path)
    global_step = ckpt["global_step"] if "global_step" in ckpt else None
    ckpt = ckpt["EMA_generator"]

    g = build_generator(cfg)
    g.eval()
    g.load_state_dict(ckpt)
    print(f"Generator loaded, num parameters: {utils.num_parameters(g)/1e6}M")
    return g, global_step


def build_trained_discriminator(cfg, global_step=None):
    ckpt_path = get_closest_ckpt(cfg.checkpoint_dir, global_step)
    ckpt = load_checkpoint(ckpt_path)["D"]

    d = build_discriminator(cfg)
    d.load_state_dict(ckpt)
    return d, ckpt_path


class TruncationStrategy(enum.Enum):
    NONE = None
    W_CLAMP = "W_CLAMP"
    W_INTERPOLATE = "W_INTERPOLATE"
    Z_CLAMP = "Z_CLAMP"
    Z_INTERPOLATE = "Z_INTERPOLATE"


def sample_from_G(batch, G, truncation_strategy: TruncationStrategy, truncation_level: float):
    z = torch.randn((batch["img"].shape[0], G.z_channels), device=batch["img"].device)
    if truncation_strategy in [TruncationStrategy.W_CLAMP, TruncationStrategy.W_INTERPOLATE] and not utils.has_intermediate_latent(G):
        logger.warn_once("Setting truncation strategy to Z_CLAMP as the generator has no style net.")
        truncation_strategy = TruncationStrategy.Z_CLAMP
    if truncation_level is None:
        return G(**batch, z=z)
    if truncation_strategy in (TruncationStrategy.W_CLAMP, TruncationStrategy.W_INTERPOLATE):
        w_avg = G.style_net.E_map.get_average_w().repeat(batch["img"].shape[0], 1)
        w = G.style_net.E_map.map_network(z)
        if truncation_strategy == TruncationStrategy.W_INTERPOLATE:
            assert abs(truncation_level) <= 1
            w = w_avg.lerp(w.float(), truncation_level)
        if truncation_strategy == TruncationStrategy.W_CLAMP:
            assert truncation_level >= 0
            w = w.clamp(w_avg-truncation_level, w_avg+truncation_level)
        z = None
    elif truncation_strategy in (TruncationStrategy.Z_CLAMP, TruncationStrategy.Z_INTERPOLATE):
        w = None
        if truncation_strategy == TruncationStrategy.Z_INTERPOLATE:
            z = z*truncation_level
            assert abs(truncation_level) <= 1
        if truncation_strategy == TruncationStrategy.Z_CLAMP:
            assert truncation_level >= 0
            z = z.clamp(-truncation_level, truncation_level)
    else:
        z, w = None, None
    
    return G(**batch, w=w, z=z)
