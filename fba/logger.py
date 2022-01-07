import torchvision
import pathlib
import torch
import numpy as np
import math
import logging
import collections
try:
    import wandb
except ImportError:
    logging.warning("Could not import wandb")
from PIL import Image
from fba import utils


class DefaultLogger:

    def __init__(self) -> None:
        pass

    def log_dictionary(self, dictionary: dict, log_level=logging.DEBUG, commit=True):
        for tag, value in dictionary.items():
            log(log_level, f"{tag}: {value}")

    def watch_models(self, models):
        print("LOGGER NOT INITIALIZED, tried to watch model.")

    def save_images(self, tag, images,
                    nrow=10,
                    denormalize=True):
        if denormalize:
            images = utils.denormalize_img(images)
        imdir = image_dir
        filename = f"{global_step}.jpg"

        filepath = imdir.joinpath(tag, filename)

        grid = torchvision.utils.make_grid(images, nrow=nrow, padding=0)
        grid = (grid * 255 + 0.5).clamp(0, 255).permute(1, 2, 0)
        grid = grid.to("cpu", torch.uint8).numpy()
        filepath.parent.mkdir(exist_ok=True, parents=True)
        info(f"Saved images to: {filepath}")
        im = Image.fromarray(grid)
        im.save(filepath)
        return im

    def finish(self):
        pass


class TensorboardLogger(DefaultLogger):

    def __init__(self, cfg) -> None:
        super().__init__()
        save_dir = cfg.output_dir.joinpath("tensorboard")
        self.writer = torch.utils.tensorboard.SummaryWriter(save_dir)
        info(f"Logging on tensorboard started. Tensorboard logs saved to {save_dir}")

    def log_dictionary(self, dictionary: dict, log_level, commit):
        super().log_dictionary(dictionary, log_level, commit)
        global_step = dictionary.pop("global_step")
        self.writer.add_scalars("", dictionary, global_step=global_step)

    def save_images(self, tag, *args, **kwargs):
        im = super().save_images(tag, *args, **kwargs)
        tag = f"images/{tag}"
        self.writer.add_image(tag, np.array(im))

    def watch_models(self, models):
        for model in models:
            info(str(model))

    def finish(self):
        self.writer.flush()
        self.writer.close()


class WandbLogger(DefaultLogger):

    def __init__(self, cfg, resume: bool, reinit: bool) -> None:
        super().__init__()
        self.run = wandb.init(
            project=cfg.project,
            dir=str(cfg.output_dir),
            name=str(cfg.filename),
            group=str(cfg.output_dir.parent),
            config=dict(cfg),
            id=get_run_id(cfg.output_dir, resume),
            resume=resume,
            reinit=reinit)

    def log_dictionary(self, dictionary: dict, log_level, commit):
        super().log_dictionary(dictionary, log_level, commit)
        wandb.log(dictionary, commit=commit)

    def save_images(self, tag, *args, **kwargs):
        im = super().save_images(tag, *args, **kwargs)
        tag = f"images/{tag}"
        wandb.log({tag: wandb.Image(im), "global_step": global_step})

    def watch_models(self, models):
        for model in models:
            info(str(model))
#            wandb.watch(model, log="parameters", log_graph=True)

    def finish(self):
        self.run.finish()
        wandb.finish()


INFO = logging.INFO
WARN = logging.WARN
DEBUG = logging.DEBUG

global_step = 0
image_dir = None
logger_backend = DefaultLogger()
logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]%(message)s")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)


def get_run_id(output_dir: pathlib.Path, resume):
    run_id_path = output_dir.joinpath("wandb_id.txt")
    if run_id_path.is_file() and resume:
        with open(run_id_path, "r") as fp:
            uid = str(fp.readlines()[0])
            return uid
    assert not resume, f"Run ID not found in: {run_id_path}"
    uid = wandb.util.generate_id()
    with open(run_id_path, "w") as fp:
        fp.write(str(uid))
    return uid


def init(cfg, resume, reinit=False):
    if utils.rank() != 0:
        return
    global image_dir, rootLogger, logger_backend
    if cfg.logger_backend == "wandb":
        logger_backend = WandbLogger(cfg, resume, reinit)
    elif cfg.logger_backend == "tensorboard":
        logger_backend = TensorboardLogger(cfg)
    elif cfg.logger_backend == "none":
        return
    else:
        raise ValueError(
            f"Not supported logger backend. Was: {cfg.logger_backend}, has to be either: [wandb, tensorboard, none]")
    image_dir = pathlib.Path(cfg.output_dir, "generated_data")
    image_dir.joinpath("validation").mkdir(exist_ok=True, parents=True)
    image_dir.joinpath("transition").mkdir(exist_ok=True, parents=True)
    filepath = pathlib.Path(cfg.output_dir, "train.log")
    fileHandler = logging.FileHandler(filepath)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)


def update_global_step(val):
    # Should generally not be called unless you know what you're doing!
    global global_step
    global_step = val


def log_variable(tag, value, log_level=logging.DEBUG):
    if utils.rank() != 0:
        return
    if math.isnan(value):
        rootLogger.debug(f"Tried to log nan/inf for tag={tag}")
        return
    value = float(value)
    log_dictionary({tag: value}, commit=False, log_level=log_level)


def log_dictionary(dictionary: dict, log_level=logging.DEBUG, commit=True):
    if utils.rank() != 0:
        return
    dictionary["global_step"] = global_step
    logger_backend.log_dictionary(dictionary, log_level, commit)


def log_model(models):
    if utils.rank() != 0:
        return
    logger_backend.watch_models(models)


def finish():
    if utils.rank() != 0:
        return

    logger_backend.finish()


def save_images(tag, images, nrow=10, denormalize=True):
    if utils.rank() != 0:
        return
    logger_backend.save_images(tag, images, nrow, denormalize)


def info(text):
    if utils.rank() != 0:
        return
    log(logging.INFO, text)


def log(log_level, text):
    if utils.rank() != 0:
        return

    text = f"[{global_step:7d}]: {text}"
    rootLogger.log(log_level, text)


def warn(text, *args):
    if utils.rank() != 0:
        return

    if len(args) > 0:
        text = text + " " + " " .join(args)
    log(logging.WARN, text)

