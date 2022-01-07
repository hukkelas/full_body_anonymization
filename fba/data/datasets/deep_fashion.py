from typing import Callable, Optional, Union
from fba import logger
import torchvision
import torch
import pathlib
import numpy as np
from .build import DATASET_REGISTRY
from fba.utils.utils import cache_embed_stats


@DATASET_REGISTRY.register_module
class DeepFashion(torch.utils.data.Dataset):

    def __init__(self,
                 dirpath: Union[str, pathlib.Path],
                 transform: Optional[Callable],
                 subset_split_file: str,
                 **kwargs):
        dirpath = pathlib.Path(dirpath)
        self.dirpath = dirpath
        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform
        assert self.dirpath.is_dir(),\
            f"Did not find dataset at: {dirpath}"

        self.image_paths, self.embedding_paths = self._load_impaths(subset_split_file)
        self.embed_map = torch.from_numpy(np.load(self.dirpath.joinpath("embed_map.npy")))
        cache_embed_stats(self.embed_map)
        logger.info(
            f"Dataset loaded from: {dirpath}. Number of samples:{len(self)}")

    def _load_impaths(self, subset_split_file):
        with open(self.dirpath.joinpath(subset_split_file), "r") as fp:
            impaths = [pathlib.Path(x.strip()) for x in fp.readlines()]
        embedding_paths = []
        image_paths = []
        for impath in impaths:
            embedding_path = self.dirpath.joinpath("cse_annotation", *impath.parts[1:-1], impath.stem + ".npy")
            if not embedding_path.is_file():
                continue
            embedding_paths.append(embedding_path)
            assert embedding_path.is_file(), embedding_path
            p = pathlib.Path(str(self.dirpath.joinpath(impath).absolute()).strip())
            assert p.is_file(), p
            image_paths.append(self.dirpath.joinpath(impath))
        return image_paths, embedding_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        im = torchvision.io.read_image(str(self.image_paths[idx]))
        vertices, mask, border = np.split(np.load(self.embedding_paths[idx]), 3, axis=-1)
        vertices = torch.from_numpy(vertices.squeeze()).long()
        mask = torch.from_numpy(mask.squeeze()).float()
        border = torch.from_numpy(border.squeeze()).float()[None]
        batch = {
            "img": im,
            "vertices": vertices,
            "mask": torch.zeros_like(mask),
            "embed_map": self.embed_map,
            "e_area": 1 - mask - border
        }
        return self.transform(batch)
