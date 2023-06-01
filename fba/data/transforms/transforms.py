from typing import  Dict, List
import torchvision
import torch
from .build import TRANSFORM_REGISTRY
import torchvision.transforms.functional as F
from .functional import hflip, one_hot_semantic, vertx2semantic




@TRANSFORM_REGISTRY.register_module
class RandomHorizontalFlip(torch.nn.Module):

    def __init__(self, p: float, **kwargs):
        super().__init__()
        self.flip_ratio = p
        if self.flip_ratio is None:
            self.flip_ratio = 0.5
        assert 0 <= self.flip_ratio <= 1

    def forward(self, container: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if torch.rand(1) > self.flip_ratio:
            return container
        return hflip(container)


@TRANSFORM_REGISTRY.register_module
class FlattenLandmark(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        return

    def forward(self, container: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "landmarks" in container,\
            f"Did not find landmarks in container. {container.keys()}"
        landmarks_XY = container["landmarks"]
        landmarks_XY = landmarks_XY.view(landmarks_XY.shape[0], -1)
        container["landmarks"] = landmarks_XY
        return container


@TRANSFORM_REGISTRY._register_module
class CenterCrop(torch.nn.Module):
    """
    Performs the transform on the image.
    NOTE: Does not transform the mask to improve runtime.
    """

    def __init__(self, size: List[int]):
        super().__init__()
        self.size = size

    def forward(self, container: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        min_size = min(container["img"].shape[1], container["img"].shape[2])
        if min_size < self.size[0]:
            container["img"] = F.center_crop(container["img"], min_size)
            container["img"] = F.resize(container["img"], self.size)
            return container
        container["img"] = F.center_crop(container["img"], self.size)
        return container


@TRANSFORM_REGISTRY._register_module
class Resize(torch.nn.Module):
    """
    Performs the transform on the image.
    NOTE: Does not transform the mask to improve runtime.
    """

    def __init__(self, size, interpolation=F.InterpolationMode.BILINEAR):
        super().__init__()
        self.size = size
        self.interpolation = interpolation

    def forward(self, container: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        container["img"] = F.resize(container["img"], self.size, self.interpolation, antialias=True)
        if "semantic_mask" in container:
            container["semantic_mask"] = F.resize(
                container["semantic_mask"], self.size, F.InterpolationMode.NEAREST)
        if "embedding" in container:
            container["embedding"] = F.resize(
                container["embedding"], self.size, self.interpolation)
        if "mask" in container:
            container["mask"] = F.resize(
                container["mask"], self.size, F.InterpolationMode.NEAREST)
        if "border" in container:
            container["border"] = F.resize(
                container["border"], self.size, F.InterpolationMode.NEAREST)

        return container


@TRANSFORM_REGISTRY._register_module
class Normalize(torch.nn.Module):
    """
    Performs the transform on the image.
    NOTE: Does not transform the mask to improve runtime.
    """

    def __init__(self, mean, std, inplace):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, container: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        container["img"] = F.normalize(container["img"], self.mean, self.std, self.inplace)
        return container


@TRANSFORM_REGISTRY._register_module
class RandomCrop(torchvision.transforms.RandomCrop):
    """
    Performs the transform on the image.
    NOTE: Does not transform the mask to improve runtime.
    """

    def forward(self, container: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        container["img"] = super().forward(container["img"])
        return container


@TRANSFORM_REGISTRY._register_module
class CreateCondition(torch.nn.Module):

    def forward(self, container: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        container["condition"] = container["img"] * container["mask"]
        return container


@TRANSFORM_REGISTRY.register_module
class ImageToTensor(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, container: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        container["img"] = torchvision.transforms.functional.to_tensor(container["img"])
        return container


@TRANSFORM_REGISTRY.register_module
class OneHotSemanticLabel(torch.nn.Module):

    def __init__(self, n_semantic: int) -> None:
        super().__init__()
        self.n_semantic = n_semantic

    def forward(self, container: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return one_hot_semantic(container, self.n_semantic)


@TRANSFORM_REGISTRY.register_module
class CreateEmbedding(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, container: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        embed_map = container["embed_map"]
        vertices = container["vertices"]
        embedding = embed_map[vertices.long()].squeeze(dim=1)

        embedding = embedding.permute(0, 3, 1, 2)
        container["embedding"] = embedding * container["E_mask"]
        return container


@TRANSFORM_REGISTRY.register_module
class Vertx2Semantic(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return vertx2semantic(batch)
