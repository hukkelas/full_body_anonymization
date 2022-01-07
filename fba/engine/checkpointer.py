import typing
import torch
from fba import logger, utils
import pathlib


def load_checkpoint_from_url(model_url: str):
    if model_url is None:
        return None
    return torch.hub.load_state_dict_from_url(
        model_url, map_location=_get_map_location())


def load_checkpoint(checkpoint_path: pathlib.Path) -> dict:
    if checkpoint_path.is_dir():
        checkpoints = get_ckpt_paths(checkpoint_path)
        if len(checkpoints) == 0:
            raise FileNotFoundError(f"No checkpoints in folder: {checkpoint_path}")
        checkpoint_path = checkpoints[-1]

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Did not find file: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=utils.get_device())
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return ckpt


def get_closest_ckpt(
        checkpoint_dir: pathlib.Path,
        global_step: int = None) -> pathlib.Path:
    paths = get_ckpt_paths(checkpoint_dir)
    if len(paths) == 0:
        raise FileNotFoundError(f"No checkpoints in folder: {checkpoint_dir}")
    if global_step is None:
        return get_ckpt_paths(checkpoint_dir)[-1]
    distances = {p: abs(int(p.stem)-global_step) for p in paths}

    paths.sort(key=lambda p: distances[p], reverse=True)
    return paths[0]


def get_ckpt_paths(checkpoint_dir: pathlib.Path) -> typing.List[pathlib.Path]:
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    checkpoints.sort(key=lambda x: int(x.stem.split("_")[-1]))
    return checkpoints


def save_checkpoint(
        checkpoint_dir: pathlib.Path,
        state_dict: dict,
        best_model_step: int,
        max_keep=2,) -> None:
    """
    Args:
        checkpoint_path: path to file or folder.
    """
    checkpoint_dir.parent.mkdir(exist_ok=True, parents=True)
    previous_checkpoint_paths = get_ckpt_paths(checkpoint_dir)
    if len(previous_checkpoint_paths) > max_keep:
        if int(previous_checkpoint_paths[0].stem) == best_model_step:
            previous_checkpoint_paths[1].unlink()
        else:
            previous_checkpoint_paths[0].unlink()
    checkpoint_path = checkpoint_dir.joinpath(f"{logger.global_step}.ckpt")
    if checkpoint_path.is_file():
        return
    torch.save(state_dict, checkpoint_path)
    logger.info(f"Saved checkpoint to: {checkpoint_path}")


def checkpoint_exists(checkpoint_dir) -> bool:
    num_checkpoints = len(list(checkpoint_dir.glob("*.ckpt")))
    return num_checkpoints > 0
