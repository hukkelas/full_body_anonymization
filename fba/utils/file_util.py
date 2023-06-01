import pathlib
import typing
import numpy as np
import os
import torch
import errno
import warnings
import sys
from urllib.parse import urlparse
from PIL import Image


def find_all_files(
    directory: pathlib.Path, suffixes=["png", "jpg", "jpeg"]
) -> typing.List[pathlib.Path]:
    image_paths = []
    for suffix in suffixes:
        image_paths.extend(directory.glob(f"*.{suffix}"))
    image_paths.sort()
    return image_paths


def find_matching_files(
    new_directory: pathlib.Path, filepaths: typing.List[pathlib.Path]
) -> typing.List[pathlib.Path]:
    new_files = []
    for impath in filepaths:
        mpath = new_directory.joinpath(impath.name)
        assert mpath.is_file(), f"Did not find path: {mpath}"
        new_files.append(mpath)
    assert len(new_files) == len(filepaths)
    return new_files


def pre_process_mask(mask):
    if mask.dtype == np.uint8:
        mask = mask.astype(np.float32) / 255
    if len(mask.shape) == 3:
        assert mask.shape[-1] == 3
        mask = mask.mean(axis=-1)
    mask = (mask > 1e-7).astype(np.float32)
    return mask[:, :, None]


def read_mask(impath: pathlib.Path):
    im = Image.open(str(impath)).convert("RGB")
    im = Image.fromarray(((np.array(im) > 0)*255).astype(np.uint8))
    mask = pre_process_mask(np.array(im))
    assert mask.max() <= 1 and mask.min() >= 0
    return mask


def read_im(impath: pathlib.Path, dtype=np.uint8):
    impath = pathlib.Path(impath)
    assert impath.is_file(),\
        f"Image path is not file: {impath}"
    im = Image.open(str(impath)).convert("RGB")
    im = np.array(im)
    if dtype == np.uint8:
        return im
    assert dtype == np.float32
    im = im.astype(np.float32) / 255
    assert im.max() <= 1 and im.min() >= 0
    return im


def download_file(url, progress=True, check_hash=False, file_name=None, subdir=None):
    r"""Downloads and caches file to TORCH CACHE

    Args:
        url (string): URL of the object to download
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Default: True
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Default: False
        file_name (string, optional): name for the downloaded file. Filename from `url` will be used if not set.

    Example:
        >>> state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')

    """
    # Issue warning to move data if old env is set
    if os.getenv('TORCH_MODEL_ZOO'):
        warnings.warn('TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')

    hub_dir = torch.hub.get_dir()
    model_dir = os.path.join(hub_dir, 'checkpoints')

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    if subdir is not None:
        filename = os.path.join(subdir, filename)
    cached_file = pathlib.Path(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            r = torch.hub.HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        cached_file.parent.mkdir(exist_ok=True, parents=True)
        torch.hub.download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    return cached_file

def torch_load_from_url(url, map_location=None, **kwargs):
    filepath = download_file(url, **kwargs)
    return torch.load(filepath, map_location=map_location)

def is_image(impath: pathlib.Path):
    return impath.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp", ".webp"]


def is_video(impath: pathlib.Path):
    return impath.suffix.lower() in [".mp4", ".webm", ".avi"]

