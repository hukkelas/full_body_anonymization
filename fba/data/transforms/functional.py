import torchvision.transforms.functional as F
from typing import  Dict
import torch
from fba.utils.file_util import download_file
import pickle


def get_symmetry_transform(symmetry_url):
    file_name = download_file(symmetry_url)
    with open(file_name, "rb") as fp:
        symmetry = pickle.load(fp)
    return symmetry["vertex_transforms"]


hflip_handled_cases = set(["landmarks", "img", "mask", "border", "semantic_mask", "vertices", "e_area", "embed_map", "condition", "embedding", "vertx2cat"])
symmetry_transform = torch.from_numpy(get_symmetry_transform("https://dl.fbaipublicfiles.com/densepose/meshes/symmetry/symmetry_smpl_27554.pkl")).long()


def hflip(container: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    container["img"] = F.hflip(container["img"])
    if "condition" in container:
        container["condition"] = F.hflip(container["condition"])
    if "embedding" in container:
        container["embedding"] = F.hflip(container["embedding"])
    assert all([key in hflip_handled_cases for key in container]), container.keys()
    if "landmarks" in container:
        landmarks_XY = container["landmarks"]
        landmarks_XY[:, :, 0] = 1 - landmarks_XY[:, :, 0]
        container["landmarks"] = landmarks_XY
    if "mask" in container:
        container["mask"] = F.hflip(container["mask"])
    if "border" in container:
        container["border"] = F.hflip(container["border"])
    if "semantic_mask" in container:
        container["semantic_mask"] = F.hflip(container["semantic_mask"])
    if "vertices" in container:
        container["vertices"] = F.hflip(container["vertices"])
        symmetry_transform_ = symmetry_transform.to(container["vertices"].device)
        container["vertices"] = symmetry_transform_[container["vertices"].long()]
    if "e_area" in container:
        container["e_area"] = F.hflip(container["e_area"])
    return container


def vertx2semantic(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    vertx2cat = batch["vertx2cat"]
    vertices = batch["vertices"]
    semantic = vertx2cat[vertices.long()]
    semantic = semantic.view(batch["mask"].shape[0], 1, vertices.shape[-2], vertices.shape[-1])
    semantic *= (1-batch["mask"])
    semantic[batch["border"].bool()] = 1
    batch["semantic_mask"] = semantic
    return batch


def one_hot_semantic(container: Dict[str, torch.Tensor], n_semantic: int) -> Dict[str, torch.Tensor]:
    if len(container["img"].shape) == 3:
        _, H, W = container["img"].shape
        semantic_mask = torch.zeros(
            (n_semantic, H, W), dtype=torch.float32,
            device=container["img"].device)
        semantic_mask.scatter_(0, container["semantic_mask"].long(), 1.0)
    else:
        bs, _, H, W = container["img"].shape
        semantic_mask = torch.zeros(
            (bs, n_semantic, H, W), dtype=torch.float32,
            device=container["img"].device)
        semantic_mask.scatter_(1, container["semantic_mask"].long(), 1.0)
    container["semantic_mask"] = semantic_mask
    return container
