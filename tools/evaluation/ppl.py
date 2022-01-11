import click
import numpy as np
import torch
from fba import utils, logger
from fba.config import Config
from fba.data import build_dataloader_val
from fba.infer import build_trained_generator
from torch_fidelity.helpers import get_kwarg, vassert, vprint
from torch_fidelity.utils import sample_random, batch_interp, create_sample_similarity

KEY_METRIC_PPL_RAW = 'perceptual_path_length_raw'
KEY_METRIC_PPL_MEAN = 'perceptual_path_length_mean'
KEY_METRIC_PPL_STD = 'perceptual_path_length_std'

def slerp(a, b, t):
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    d = (a * b).sum(dim=-1, keepdim=True)
    p = t * torch.acos(d)
    c = b - d * a
    c = c / c.norm(dim=-1, keepdim=True)
    d = a * torch.cos(p) + c * torch.sin(p)
    d = d / d.norm(dim=-1, keepdim=True)
    return d


@torch.no_grad()
def calculate_ppl(
    dataloader, 
    G,
    space,
    **kwargs):
    """
    Inspired by https://github.com/NVlabs/stylegan/blob/master/metrics/perceptual_path_length.py
    """
    verbose = get_kwarg('verbose', kwargs)
    epsilon = get_kwarg('ppl_epsilon', kwargs)
    interp = get_kwarg('ppl_z_interp_mode', kwargs)
    reduction = get_kwarg('ppl_reduction', kwargs)
    similarity_name = get_kwarg('ppl_sample_similarity', kwargs)
    sample_similarity_resize = get_kwarg('ppl_sample_similarity_resize', kwargs)
    sample_similarity_dtype = get_kwarg('ppl_sample_similarity_dtype', kwargs)
    discard_percentile_lower = get_kwarg('ppl_discard_percentile_lower', kwargs)
    discard_percentile_higher = get_kwarg('ppl_discard_percentile_higher', kwargs)

    vassert(type(epsilon) is float and epsilon > 0, 'Epsilon must be a small positive floating point number')
    vassert(reduction in ('none', 'mean'), 'Reduction must be one of [none, mean]')
    vassert(discard_percentile_lower is None or 0 < discard_percentile_lower < 100, 'Invalid percentile')
    vassert(discard_percentile_higher is None or 0 < discard_percentile_higher < 100, 'Invalid percentile')
    if discard_percentile_lower is not None and discard_percentile_higher is not None:
        vassert(0 < discard_percentile_lower < discard_percentile_higher < 100, 'Invalid percentiles')

    sample_similarity = create_sample_similarity(
        similarity_name,
        sample_similarity_resize=sample_similarity_resize,
        sample_similarity_dtype=sample_similarity_dtype,
        cuda=torch.cuda.is_available(),
        **kwargs
    )

    rng = np.random.RandomState(get_kwarg('rng_seed', kwargs))
    num_samples = len(dataloader)*dataloader.batch_size
    if G.class_specific_z:
        lat_e0 = sample_random(rng, (num_samples, G.z_channels*G.semantic_nc), "normal")
        lat_e1 = sample_random(rng, (num_samples, G.z_channels*G.semantic_nc), "normal")
    else:
        lat_e0 = sample_random(rng, (num_samples, G.z_channels), "normal")
        lat_e1 = sample_random(rng, (num_samples, G.z_channels), "normal")
    if space == "Z":
        lat_e1 = batch_interp(lat_e0, lat_e1, epsilon, interp)

    distances = []
    for it, batch in enumerate(utils.tqdm_(dataloader, desc="Perceptual Path Length")):
        start = it*dataloader.batch_size
        end = start + batch["img"].shape[0]
        batch = {k: torch.cat((x, x)) for k, x in batch.items()}
        batch_lat_e0 = utils.to_cuda(lat_e0[start:end])
        batch_lat_e1 = utils.to_cuda(lat_e1[start:end])
        if G.class_specific_z:
            batch_lat_e0 = batch_lat_e0.view(-1, G.semantic_nc, G.z_channels)
            batch_lat_e1 = batch_lat_e1.view(-1, G.semantic_nc, G.z_channels)
        if space == "W":
            w0, w1 = G.style_net.E_map.map_network(torch.cat((batch_lat_e0, batch_lat_e1))).chunk(2)
            w1 = w0.lerp(w1, epsilon) # PPL end
            img = G(**batch, w=torch.cat((w0, w1)))["img"]
        else:
            assert space == "Z"
            img = G(**batch, z=torch.cat((batch_lat_e0, batch_lat_e1)))["img"]

        rgb1, rgb2 = (utils.denormalize_img(img)*255).byte().chunk(2)
        

        sim = sample_similarity(rgb1, rgb2)
        dist_lat_e01 = sim / (epsilon ** 2)
        distances.append(dist_lat_e01.cpu().numpy())

    distances = np.concatenate(distances, axis=0)

    cond, lo, hi = None, None, None
    if discard_percentile_lower is not None:
        lo = np.percentile(distances, discard_percentile_lower, interpolation='lower')
        cond = lo <= distances
    if discard_percentile_higher is not None:
        hi = np.percentile(distances, discard_percentile_higher, interpolation='higher')
        cond = np.logical_and(cond, distances <= hi)
    if cond is not None:
        distances = np.extract(cond, distances)

    out = {
        KEY_METRIC_PPL_MEAN: float(np.mean(distances)),
        KEY_METRIC_PPL_STD: float(np.std(distances))
    }
    if reduction == 'none':
        out[KEY_METRIC_PPL_RAW] = distances

    vprint(verbose, f'Perceptual Path Length: {out[KEY_METRIC_PPL_MEAN]} ± {out[KEY_METRIC_PPL_STD]}')

    return out

@click.command()
@click.argument("config_path")
def main(config_path):
    cfg = Config.fromfile(config_path)
    space = "Z"
    if hasattr(cfg.generator.style_cfg, "w_mapper"):
        cfg.generator.style_cfg.w_mapper.normalize_z = False
        if cfg.generator.style_cfg.w_mapper.input_z:
            space = "W"
    print("USING SPACE:", space)
    G, global_step = build_trained_generator(cfg)
    cfg.data_val.loader.batch_size = 4
    dl = build_dataloader_val(cfg)
    
    ppl = calculate_ppl(dl, G, space=space)
    ppl = {f"ppl/{k}": v for k,v in ppl.items()}
    logger.init(cfg, resume=True)
    logger.update_global_step(global_step)
    logger.log_dictionary(ppl)
    logger.finish()


if __name__ == "__main__":
    main()
