from .transforms import build_transforms
from .utils import DataPrefetcher, InfiniteSampler
from .datasets import build_dataset
import torch
from fba import utils
from torch.utils.data._utils.collate import default_collate


def get_dataloader(cfg, is_train: bool):
    imsize = cfg.imsize
    if is_train:
        cfg_data = cfg.data_train
    else:
        cfg_data = cfg.data_val

    gpu_transform = build_transforms(
        cfg_data.image_gpu_transforms, imsize, cfg.jit_transform)
    cpu_transform = build_transforms(cfg_data.cpu_transforms, imsize, False)
    dataset = build_dataset(cfg_data.dataset, imsize, cpu_transform, is_train=is_train)
    sampler = None
    additional_kwargs = {}
    if is_train:
        sampler = InfiniteSampler(
            dataset, rank=utils.rank(),
            num_replicas=utils.world_size(), 
            **cfg_data.sampler)
    elif utils.world_size() > 1:
        sampler = torch.utils.data.DistributedSampler(
            dataset, **cfg_data.sampler, num_replicas=utils.world_size(), rank=utils.rank())
        additional_kwargs["drop_last"] = cfg_data.sampler["drop_last"]
    else:
        additional_kwargs.update(cfg_data.sampler)
    dataloader = torch.utils.data.DataLoader(
        dataset, sampler=sampler, collate_fn=collate_fn,
        **cfg_data.loader,**additional_kwargs,
    )
    embed_map = None
    if hasattr(dataset, "embed_map"):
        embed_map = dataset.embed_map
    dataloader = DataPrefetcher(
        dataloader,
        image_gpu_transforms=gpu_transform,
        embed_map=embed_map
    )
    return dataloader


def collate_fn(batch):
    elem = batch[0]
    ignore_keys = set(["embed_map", "vertx2cat"])
    batch_ = {key: default_collate([d[key] for d in batch]) for key in elem if key not in ignore_keys} 
    if "embed_map" in elem:
        batch_["embed_map"] = elem["embed_map"]
    if "vertx2cat" in elem:
        batch_["vertx2cat"] = elem["vertx2cat"]
    return batch_


def build_dataloader_train(cfg):
    return get_dataloader(cfg, is_train=True)


def build_dataloader_val(cfg):
    return get_dataloader(cfg, is_train=False)

