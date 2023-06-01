import pickle
import torch
import time
import torch_fidelity
from pathlib import Path
from fba import utils
from .lpips import SampleSimilarityLPIPS
from torch_fidelity.defaults import DEFAULTS as trf_defaults
from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric
from torch_fidelity.utils import create_feature_extractor
lpips_model = None
fid_model = None

@torch.no_grad()
def mse(images1: torch.Tensor, images2: torch.Tensor) -> torch.Tensor:
    se = (images1 - images2) ** 2
    se = se.view(images1.shape[0], -1).mean(dim=1)
    return se

@torch.no_grad()
def psnr(images1: torch.Tensor, images2: torch.Tensor) -> torch.Tensor:
    mse_ = mse(images1, images2)
    psnr = 10 * torch.log10(1 / mse_)
    return psnr

@torch.no_grad()
def lpips(images1: torch.Tensor, images2: torch.Tensor) -> torch.Tensor:
    return _lpips_w_grad(images1, images2)


def _lpips_w_grad(images1: torch.Tensor, images2: torch.Tensor) -> torch.Tensor:
    global lpips_model
    if lpips_model is None:
        lpips_model = utils.to_cuda(SampleSimilarityLPIPS())
    
    images1 = images1.mul(255)
    images2 = images2.mul(255)
    with torch.cuda.amp.autocast(utils.AMP()):
        dists = lpips_model(images1, images2)[0].view(-1)
    return dists




@torch.no_grad()
@torch.inference_mode()
def compute_metrics_iteratively(
        dataloader, generator,
        cache_directory,
        truncation_value=999999,
        autocast_fp16=True,
        include_two_fake = False
        ):
    """
    Args:
        n_samples (int): Creates N samples from same image to calculate stats
    """
    global lpips_model, fid_model
    if lpips_model is None:
        lpips_model = utils.to_cuda(SampleSimilarityLPIPS())
    if fid_model is None:
        fid_model = create_feature_extractor(
            trf_defaults["feature_extractor"], [trf_defaults["feature_layer_fid"]], cuda=False)
        fid_model = utils.to_cuda(fid_model)
    cache_directory = Path(cache_directory)
    start_time = time.time()
    lpips_total = torch.tensor(0, dtype=torch.float32, device=utils.get_device())
    diversity_total = torch.zeros_like(lpips_total)
    assert dataloader.drop_last
    fid_cache_path = cache_directory.joinpath("fid_stats.pkl")
    has_fid_cache = fid_cache_path.is_file()
    if not has_fid_cache:
        fid_features_real = torch.zeros(len(dataloader)*dataloader.batch_size, 2048, dtype=torch.float32, device=utils.get_device())
    fid_features_fake = torch.zeros(len(dataloader)*dataloader.batch_size * (1+include_two_fake), 2048, dtype=torch.float32, device=utils.get_device())
    for batch_idx, batch in enumerate(utils.tqdm_(dataloader, desc="Validating on dataset.")):
        sidx = batch_idx * dataloader.batch_size
        eidx = (1+batch_idx) * dataloader.batch_size
        assert batch["img"].shape[0] == dataloader.batch_size, batch["img"].shape[0]
        with torch.cuda.amp.autocast(autocast_fp16 and utils.AMP()):
            fakes1 = generator(**batch, truncation_value=truncation_value)["img"]
            fakes2 = generator(**batch, truncation_value=truncation_value)["img"]
            real_data = batch["img"]
            fakes1 = utils.denormalize_img(fakes1).mul(255)
            fakes2 = utils.denormalize_img(fakes2).mul(255)
            real_data = utils.denormalize_img(real_data).mul(255)
            lpips_1, real_lpips_feats, fake1_lpips_feats = lpips_model(real_data, fakes1)
            fake2_lpips_feats = lpips_model.get_feats(fakes2)
            lpips_2 = lpips_model.lpips_from_feats(real_lpips_feats, fake2_lpips_feats)

            lpips_total += lpips_1.mean().add(lpips_2.mean()).div(2)
            diversity_total += lpips_model.lpips_from_feats(fake1_lpips_feats, fake2_lpips_feats).mean()
            if not has_fid_cache:
                fid_features_real[sidx:eidx] = fid_model(real_data.byte())[0]
            sidx = sidx*(1+include_two_fake)
            fid_features_fake[sidx:sidx+dataloader.batch_size] = fid_model(fakes1.byte())[0]
            if include_two_fake:
                fid_features_fake[sidx+dataloader.batch_size:sidx+dataloader.batch_size*2] = fid_model(fakes2.byte())[0]
    if has_fid_cache:
        with open(fid_cache_path, "rb") as fp:
            fid_stat_real = pickle.load(fp)
    else:
        fid_stat_real = fid_features_to_statistics(utils.gather_tensors(fid_features_real).cpu())
        cache_directory.mkdir(exist_ok=True, parents=True)
        with open(fid_cache_path, "wb") as fp:
            pickle.dump(fid_stat_real, fp)
    fid_stat_fake = fid_features_to_statistics(utils.gather_tensors(fid_features_fake).cpu())
    fid_ =  fid_statistics_to_metric(fid_stat_real, fid_stat_fake, verbose=False)["frechet_inception_distance"]
    to_return = {
        "lpips": (lpips_total / len(dataloader)),
        "lpips_diversity": (diversity_total / len(dataloader))
    }
    for key in to_return.keys():
        utils.all_reduce(to_return[key], torch.distributed.ReduceOp.SUM)
        to_return[key] /= utils.world_size()
    to_return = {k: v.cpu().item() for k, v in to_return.items()}
    to_return["fid"] = fid_
    to_return["validation_time_s"] = time.time() - start_time
    return to_return
