import pathlib
import torchvision
from PIL import Image
import numpy as np
import torch
import tqdm
import random
import os
from torchvision.transforms.functional import InterpolationMode, resize


def get_device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{rank()}")
    return torch.device("cpu")


def denormalize_img(image, mean=0.5, std=0.5):
    image = image * std + mean
    image = torch.clamp(image.float(), 0, 1)
    image = (image * 255)
    image = torch.round(image)
    return image / 255


def num_parameters(module: torch.nn.Module):
    count = 0
    for p in module.parameters():
        count += np.prod(p.shape)
    return count


def _to_cuda(element):
    return element.to(get_device(), non_blocking=True)


def to_cuda(elements):
    if isinstance(elements, tuple) or isinstance(elements, list):
        return [_to_cuda(x) for x in elements]
    return _to_cuda(elements)


def AMP() -> bool:
    if "AMP_ENABLED" not in os.environ:
        print("Setting AMP to False")
        return False
    if os.environ["AMP_ENABLED"] in ["1", None]:
        return True
    return False


@torch.no_grad()
def image2np(images, to_uint8=False, denormalize=False):
    single_image = False
    if len(images.shape) == 3:
        single_image = True
        images = images[None]
    if denormalize:
        images = denormalize_img(images)
    if images.dtype != torch.uint8:
        images = images.clamp(0, 1)
    images = images.detach().cpu().numpy()

    images = np.moveaxis(images, 1, -1)
    if to_uint8:
        images = (images * 255).astype(np.uint8)
    if single_image:
        return images[0]
    return images


def im2torch(im, cuda=False, normalize=True, to_float=True):
    assert len(im.shape) in [3, 4]
    single_image = len(im.shape) == 3
    if im.dtype == np.uint8 and to_float:
        im = im.astype(np.float32)
        im /= 255
    if single_image:
        im = np.rollaxis(im, 2)
        im = im[None, :, :, :]
    else:
        im = np.moveaxis(im, -1, 1)
    image = torch.from_numpy(im).contiguous()
    if cuda:
        image = to_cuda(image)
    if to_float:
        assert image.min() >= 0.0 and image.max() <= 1.0
        if normalize:
            image = image * 2 - 1
    return image

_seed = 0
def set_seed(seed: int):
    global _seed
    _seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_seed():
    global _seed
    return _seed


def gaussian_kl(mu, logvar):
    """
    KL divergence to a normal distribution N(0, 1)
    """
    return - 0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)


def mask2torch(mask: np.ndarray, cuda=True):
    assert mask.max() <= 1 and mask.min() >= 0
    mask = mask.squeeze()
    single_mask = len(mask.shape) == 2
    assert len(mask.shape) in [2, 3], (mask.shape)
    if single_mask:
        mask = mask[None]
    mask = mask[:, None, :, :]
    mask = torch.from_numpy(mask)
    if cuda:
        mask = to_cuda(mask)
    return mask.float()


def zero_grad(model):
    """
    Reduce overhead of optimizer.zero_grad (read+write).
    """
    for param in model.parameters():
        param.grad = None


def set_requires_grad(module: torch.nn.Module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad = requires_grad


def forward_D_fake(batch, fake_img, discriminator):
    fake_batch = {k: v for k, v in batch.items() if k != "img"}
    fake_batch["img"] = fake_img
    return discriminator(**fake_batch)


_world_size = 1
_rank = 0


def rank() -> int:
    return _rank


def world_size() -> int:
    return _world_size


def set_world_size_and_rank(rank, world_size):
    global _world_size, _rank
    _rank = rank
    _world_size = world_size


def gather_tensors(tensor):
    if world_size() <= 1:
        return tensor
    output = [tensor.clone() for _ in range(world_size())]
    torch.distributed.all_gather(tensor=tensor, tensor_list=output)
    return torch.cat(output)


def all_reduce(tensor, op):
    if world_size() <= 1:
        return None
    torch.distributed.all_reduce(tensor, op)


def tqdm_(iterator, *args, **kwargs):
    if rank() == 0:
        return tqdm.tqdm(iterator, *args, **kwargs)
    return iterator

def trange_(iterator, *args, **kwargs):
    if rank() == 0:
        return tqdm.trange(iterator, *args, **kwargs)
    return iterator


def cut_pad_resize(x: torch.Tensor, bbox, target_shape):
    """
        Crops or pads x to fit in the bbox and resize to target shape.
    """
    C, H, W = x.shape
    x0, y0, x1, y1 = bbox

    if y0 > 0 and x1 > 0 and x1 <= W and y1 <= H:
        new_x = x[:, y0:y1, x0:x1]
    else:
        new_x = torch.zeros(((C, y1-y0, x1-x0)), dtype=x.dtype, device=x.device)
        y0_t = max(0, -y0)
        y1_t = min(y1-y0, (y1-y0)-(y1-H))
        x0_t = max(0, -x0)
        x1_t = min(x1-x0, (x1-x0)-(x1-W))
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(x1, W)
        y1 = min(y1, H)
        new_x[:, y0_t:y1_t, x0_t:x1_t] = x[:, y0:y1, x0:x1]
    if x1 - x0 == target_shape[1] and y1 - y0 == target_shape[0]:
        return new_x
    if x.dtype == torch.bool:
        new_x = resize(new_x.float(), target_shape, interpolation=InterpolationMode.NEAREST) > 0.5
    elif x.dtype == torch.float32:
        new_x = resize(new_x, target_shape, interpolation=InterpolationMode.BILINEAR, antialias=True)
    elif x.dtype == torch.uint8:
        new_x = resize(new_x.float(), target_shape, interpolation=InterpolationMode.BILINEAR, antialias=True).byte()
    else:
        raise ValueError(f"Not supported dtype: {x.dtype}")
    return new_x


def masks_to_boxes(segmentation: torch.Tensor):
    assert len(segmentation.shape) == 3
    x = segmentation.any(dim=1).byte() # Compress rows
    x0 = x.argmax(dim=1)

    x1 = segmentation.shape[2] - x.flip(dims=(1,)).argmax(dim=1)
    y = segmentation.any(dim=2).byte()
    y0 = y.argmax(dim=1)
    y1 = segmentation.shape[1] - y.flip(dims=(1,)).argmax(dim=1)
    return torch.stack([x0, y0, x1, y1], dim=1)
