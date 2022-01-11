import shutil
from PIL import Image
import numpy as np
import torch
from fba import utils
from fba.config import Config
from fba.anonymizer import build_anonymizer
from detectron2.data.detection_utils import  _apply_exif_orientation
import click, pathlib, pickle
utils.set_seed(0)

def iteratively_anonymize(
        source_directory: pathlib.Path, cse_directory: pathlib.Path,
        target_directory: pathlib.Path, anonymizer, truncation_value):
    target_directory.mkdir(exist_ok=True, parents=True)
    for path in source_directory.iterdir():
        if "train2017" in path.parts: continue
        if path.is_dir():
            iteratively_anonymize(path, cse_directory.joinpath(path.name), target_directory.joinpath(path.name), anonymizer)
            continue
        if utils.is_image(path):# and "JPEGImages" in path.parts:
            cse_path = cse_directory.joinpath(path.stem + ".pickle")
            with open(cse_path, "rb") as fp:
                detections = pickle.load(fp)
            if len(detections) == 0:
                shutil.copy(path, target_directory.joinpath(path.name))
                continue
            assert len(detections) > 0
            im = Image.open(path)
            im = _apply_exif_orientation(im)
            orig_im_mode = im.mode
            im = im.convert("RGB")
            im = torch.from_numpy(np.rollaxis(im, 2))
            im = anonymizer.forward(im, detections=detections, truncation_value=truncation_value)
            im = Image.fromarray(im).convert(orig_im_mode)
            output_path = target_directory.joinpath(path.name)
            im.save(output_path, format="JPEG", optimize=False, quality=100, subsampling=0)
        else:
            shutil.copy(path, target_directory.joinpath(path.name))



@click.command()
@click.argument("config_path")
@click.argument("source_directory")
@click.argument("target_directory")
@click.option("-t", "--truncation_value", default=0, type=float)
def main(config_path, source_directory, target_directory, truncation_value):
    cfg = Config.fromfile(config_path)
    source_directory = pathlib.Path(source_directory)
    target_directory = pathlib.Path(target_directory)
    cse_directory = source_directory.parent.joinpath(source_directory.stem + "_cse_detections")
    anonymizer = build_anonymizer(cfg, detection_score_threshold=0.1)
    iteratively_anonymize(source_directory, cse_directory, target_directory, anonymizer, truncation_value)


main()
