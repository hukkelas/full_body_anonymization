import torch
import time
import torch_fidelity
from pathlib import Path
from torch_fidelity.generative_model_modulewrapper import GenerativeModelModuleWrapper
from fba import utils, logger
from .lpips import PerceptualLoss


lpips_model = None

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
        lpips_model = PerceptualLoss(
            model="net-lin", net="alex"
        )
    images1 = images1 * 2 - 1
    images2 = images2 * 2 - 1
    with torch.cuda.amp.autocast(utils.AMP()):
        dists = lpips_model(images1, images2, normalize=False).view(-1)
    return dists


@torch.no_grad()
def compute_metrics_iteratively(
        dataloader, generator, n_diversity_samples: int,
        fid_real_directory: str,
        truncation_value=999999):
    """
    Args:
        n_samples (int): Creates N samples from same image to calculate stats
    """
    assert n_diversity_samples > 0
    N_images = 0
    accumulated_metrics = {"lpips": 0, "lpips/diversity": 0}
    start_time = time.time()
    if utils.rank() == 0:
        fake_images = []
        real_images = []
    for batch in utils.tqdm_(dataloader, desc="Validating on dataset."):
        real_data = batch["img"]
        lpips_values = torch.zeros(
                real_data.shape[0], n_diversity_samples * 2, dtype=real_data.dtype, device=real_data.device)
        lpips_diversity = torch.zeros(
            real_data.shape[0], n_diversity_samples, dtype=real_data.dtype, device=real_data.device)

        real_data = utils.denormalize_img(real_data)
        r = utils.gather_tensors((real_data*255).byte())
        if utils.rank() == 0:
            real_images.append(r.cpu())
        for i in range(n_diversity_samples):
            with torch.cuda.amp.autocast(enabled=utils.AMP()):
                fakes1 = generator(**batch, truncation_value=truncation_value)["img"]
                fakes2 = generator(**batch, truncation_value=truncation_value)["img"]
            fakes1 = utils.denormalize_img(fakes1)
            fakes2 = utils.denormalize_img(fakes2)
            f1 = utils.gather_tensors((fakes1*255).byte()).cpu()
            f2 = utils.gather_tensors((fakes2*255).byte()).cpu()
            if utils.rank() == 0:
                fake_images.append(f1)
                fake_images.append(f2)
            lpips_diversity[:, i] = lpips(fakes1, fakes2)
            lpips_values[:, 2*i] = lpips(real_data, fakes1)
            lpips_values[:, 2*i+1] = lpips(real_data, fakes2)
        accumulated_metrics["lpips"] += lpips_values.mean(dim=1).sum()
        accumulated_metrics["lpips/diversity"] += lpips_diversity.mean(dim=1).sum()
        N_images += batch["img"].shape[0]

    for key in accumulated_metrics.keys():
        utils.all_reduce(accumulated_metrics[key], torch.distributed.ReduceOp.SUM)
        accumulated_metrics[key] /= (N_images * utils.world_size())
    to_return = {k: v.cpu().item() for k, v in accumulated_metrics.items()}

    if utils.rank() == 0:
        logger.info(f"Computing FID from {len(fake_images)*fake_images[0].shape[0]} samples")

        fid_metrics = torch_fidelity.calculate_metrics(
            input1=ImageIteratorWrapper(fake_images, ),
            input2=ImageIteratorWrapper(real_images),
            cuda=torch.cuda.is_available(),
            fid=True,
            input2_cache_name="_".join(Path(fid_real_directory).parts) + "_cached_imsize" + str(fake_images[0].shape[-1]),
            batch_size=fake_images[0].shape[0],
            input1_model_num_samples=len(fake_images)*fake_images[0].shape[0],
            input2_model_num_samples=len(real_images)*real_images[0].shape[0],
        )
        to_return["fid"] = fid_metrics["frechet_inception_distance"]
        del fid_metrics["frechet_inception_distance"]
        to_return.update(fid_metrics)
    val_time = time.time() - start_time
    to_return["validation_time_s"] = val_time
    return to_return


class ImageIteratorWrapper(GenerativeModelModuleWrapper):

    def __init__(self, images):
        super().__init__(torch.nn.Module(), 1, "normal", 0)
        self.images = images
        self.it = 0

    @torch.no_grad()
    def forward(self, z, **kwargs):
        self.it += 1
        return self.images[self.it-1].to(utils.get_device())
