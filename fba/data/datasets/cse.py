import pickle
from typing import Callable, Optional, Union
from fba import logger
import torchvision
import torch
import pathlib
import numpy as np
from .build import DATASET_REGISTRY
from fba.utils.utils import cache_embed_stats


@DATASET_REGISTRY.register_module
class CocoCSE(torch.utils.data.Dataset):


    def __init__(self,
                 dirpath: Union[str, pathlib.Path],
                 transform: Optional[Callable],
                 **kwargs):
        dirpath = pathlib.Path(dirpath)
        self.dirpath = dirpath
        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform
        assert self.dirpath.is_dir(),\
            f"Did not find dataset at: {dirpath}"
        self.image_paths, self.embedding_paths = self._load_impaths()
        self.embed_map = torch.from_numpy(np.load(self.dirpath.joinpath("embed_map.npy")))
        cache_embed_stats(self.embed_map)
        logger.info(
            f"Dataset loaded from: {dirpath}. Number of samples:{len(self)}")

    def _load_impaths(self):
        image_dir = self.dirpath.joinpath("images")
        image_paths = list(image_dir.glob("*.png"))
        image_paths.sort()
        embedding_paths = [
            self.dirpath.joinpath("embedding", x.stem + ".npy") for x in image_paths
            ]
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
            "mask": mask,
            "embed_map": self.embed_map,
            "border": border
        }
        return self.transform(batch)


@DATASET_REGISTRY.register_module
class CocoCSEWithFace(CocoCSE):

    def __init__(self,
                 dirpath: Union[str, pathlib.Path],
                 transform: Optional[Callable],
                 **kwargs):
        super().__init__(dirpath, transform, **kwargs)
        with open(self.dirpath.joinpath("face_boxes_XYXY.pickle"), "rb") as fp:
            self.face_boxes = pickle.load(fp)
        

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item["boxes_XYXY"] = self.face_boxes[self.image_paths[idx].name]
        return item


@DATASET_REGISTRY.register_module
class CocoCSESemantic(torch.utils.data.Dataset):


    def __init__(self,
                 dirpath: Union[str, pathlib.Path],
                 transform: Optional[Callable],
                 **kwargs):
        dirpath = pathlib.Path(dirpath)
        self.dirpath = dirpath
        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform
        assert self.dirpath.is_dir(),\
            f"Did not find dataset at: {dirpath}"
        self.image_paths, self.embedding_paths = self._load_impaths()
        self.vertx2cat = torch.from_numpy(np.load(self.dirpath.parent.joinpath("vertx2cat.npy")))
        self.embed_map = torch.from_numpy(np.load(self.dirpath.joinpath("embed_map.npy")))
        logger.info(
            f"Dataset loaded from: {dirpath}. Number of samples:{len(self)}")

    def _load_impaths(self):
        image_dir = self.dirpath.joinpath("images")
        image_paths = list(image_dir.glob("*.png"))
        image_paths.sort()
        embedding_paths = [
            self.dirpath.joinpath("embedding", x.stem + ".npy") for x in image_paths
            ]
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
            "mask": mask,
            "border": border,
            "vertx2cat": self.vertx2cat,
            "embed_map": self.embed_map,
        }
        return self.transform(batch)

@DATASET_REGISTRY.register_module
class CocoCSESemanticWithFace(CocoCSESemantic):

    def __init__(self,
                 dirpath: Union[str, pathlib.Path],
                 transform: Optional[Callable],
                 **kwargs):
        super().__init__(dirpath, transform, **kwargs)
        with open(self.dirpath.joinpath("face_boxes_XYXY.pickle"), "rb") as fp:
            self.face_boxes = pickle.load(fp)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item["boxes_XYXY"] = self.face_boxes[self.image_paths[idx].name]
        return item

