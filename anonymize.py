import click
import numpy as np
import tqdm
import moviepy.editor as mp
import cv2
from pathlib import Path
from PIL import Image
from fba.config import Config
from fba.anonymizer import build_anonymizer
from fba import utils, logger
from detectron2.data.detection_utils import _apply_exif_orientation

def show_video(video_path):
    video_cap = cv2.VideoCapture(str(video_path))
    while video_cap.isOpened():
        ret,  frame = video_cap.read()
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(25)
        if key == ord("q"):
            break
    video_cap.release()
    cv2.destroyAllWindows()

def anonymize_video(video_path, output_path, anonymizer, visualize: bool, max_res: int, start_time: int, save: bool, fps: int, truncation_value: float, **kwargs):
    video = mp.VideoFileClip(str(video_path))
    def process_frame(frame):
        frame = np.array(resize(Image.fromarray(frame), max_res))
        anonymized = anonymizer(frame, truncation_value=truncation_value)

        if visualize:
            cv2.imshow("frame", anonymized[:, :, ::-1])
            key = cv2.waitKey(1)
            if key == ord("q"):
                exit()
        return anonymized
    video = video.subclip(start_time,)
    if fps is not None:
        video = video.set_fps(fps)
    video = video.fl_image(process_frame)
    video.write_videofile(str(output_path))

def resize(frame: Image.Image, max_res):
    f = max(*[x/max_res for x in frame.size], 1)
    if  f  == 1:
        return frame
    new_shape = [int(x/f) for x in frame.size]
    return frame.resize(new_shape,  resample=Image.BILINEAR)


def anonymize_image(image_path, output_path, visualize: bool, anonymizer, max_res: int, save: bool, truncation_value: float, **kwargs):
    with Image.open(image_path) as im:
        im = _apply_exif_orientation(im)
        orig_im_mode = im.mode

        im = im.convert("RGB")
        im = resize(im, max_res)
    im = utils.im2torch(np.array(im), to_float=False)[0]
    im_ = anonymizer(im, truncation_value=truncation_value)
    im_ = utils.image2np(im_)
    if visualize:
        while True:
            cv2.imshow("frame", im_[:, :, ::-1])
            key = cv2.waitKey(0)
            if key == ord("q"):
                break
            elif key == ord("u"):
                im_ = utils.image2np(anonymizer(im))
        return
    im = Image.fromarray(im_).convert(orig_im_mode)
    if save:
        im.save(output_path, optimize=False, quality=100)
        print(f"Saved to: {output_path}")


def anonymize_file(input_path: Path, output_path: Path, **kwargs):
    if output_path.is_file():
        logger.warn(f"Overwriting previous file: {output_path}")
    if utils.is_image(input_path):
        anonymize_image(input_path, output_path, **kwargs)
    elif utils.is_video(input_path):
        anonymize_video(input_path, output_path, **kwargs)
    else:
        logger.info(f"Filepath not a video or image file: {input_path}")


def anonymize_directory(input_dir: Path, output_dir: Path, **kwargs):
    output_dir.mkdir(exist_ok=True, parents=True)
    for childname in tqdm.tqdm(input_dir.iterdir()):
        childpath = input_dir.joinpath(childname.name)
        output_path = output_dir.joinpath(childname.name)
        if not childpath.is_file():
            anonymize_directory(childpath, output_path, **kwargs)
        else:
            assert childpath.is_file()
            anonymize_file(childpath, output_path, **kwargs)



@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("input_path", type=click.Path(exists=True))#, help="Input path. Accepted inputs: images, videos, directories.")
@click.option("--output_path", default=None, type=click.Path())
@click.option("--model_name", default="anonymized", type=str, help="Model name/subidirectory to save image if output path not given.")
@click.option("--visualize", default=False, is_flag=True, help="Visualize the result")
@click.option("--max_res", default=1920, type=int, help="Maximum resolution  of height/wideo")
@click.option("--start_time", "--ss", default=0, type=int, help="Start time (second) for vide anonymization")
@click.option("--fps", default=None, type=int, help="FPS for anonymization")
@click.option("--save", default=False, is_flag=True)
@click.option("-t", "--truncation_value", default=0, type=click.FloatRange(0, 5), help="")
@click.option("--detection_score_threshold", default=.3, type=click.FloatRange(0, 1), help="Detection threshold")
def anonymize_path(config_path, input_path, output_path, model_name, detection_score_threshold, **kwargs):
    cfg = Config.fromfile(config_path)
    anonymizer = build_anonymizer(cfg,
        detection_score_threshold=detection_score_threshold)
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path is not None else None
    kwargs["anonymizer"] = anonymizer
    if input_path.is_dir():
        if output_path is None:
            output_path = input_path.parent.joinpath(input_path.stem + "_" + model_name)
        assert not output_path.is_file()
        anonymize_directory(input_path, output_path, **kwargs)
    else:
        if output_path is None:
            output_path = input_path.parent.joinpath(f"{input_path.stem}_anonymized{input_path.suffix}")
        anonymize_file(input_path, output_path, **kwargs)
    
    

if __name__ == "__main__":
    anonymize_path()