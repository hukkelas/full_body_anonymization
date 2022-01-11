import torch
from fba import utils
from fba.config import Config
from fba.anonymizer import build_anonymizer
from detectron2.data.detection_utils import _apply_exif_orientation
from PIL import Image
import numpy as np
import click, pathlib, pickle
utils.set_seed(0)

def iteratively_detect(source_directory:pathlib.Path, target_directory: pathlib.Path, anonymizer):
    target_directory.mkdir(exist_ok=True, parents=True)
    for path in source_directory.iterdir():
        print(path)
        if path.is_dir():
            iteratively_detect(path, target_directory.joinpath(path.name), anonymizer)
            continue
        if utils.is_image(path):
            im = Image.open(path)
            im = _apply_exif_orientation(im)
            orig_im_mode = im.mode
            im = im.convert("RGB")
            im = torch.from_numpy(np.rollaxis(np.array(im), 2))
            detections = anonymizer.detector(im)
            target_path = target_directory.joinpath(path.stem + ".pickle")

            with open(target_path, "wb") as fp:
                pickle.dump(detections, fp)



@click.command()
@click.argument("config_path")
@click.argument("source_directory")
def main(config_path, source_directory):
    cfg = Config.fromfile(config_path)
    source_directory = pathlib.Path(source_directory)
    target_directory = source_directory.parent.joinpath(source_directory.stem + "_cse_detections")
    anonymizer = build_anonymizer(cfg, detection_score_threshold=0.1)
    iteratively_detect(source_directory, target_directory, anonymizer)
    
