import gradio
import numpy as np
import torch
from PIL import Image
from fba.anonymizer import build_anonymizer
from fba import utils
from fba.config import Config
import gradio.inputs
def anonymize(img: Image.Image, truncation_value: float):
    img = img.convert("RGB")
    img = np.array(img)
    img = np.rollaxis(img, 2)
    img = torch.from_numpy(img)
    assert img.dtype == torch.uint8
    img = anonymizer(img, truncation_value=truncation_value)
    img = utils.image2np(img)
    return img

cfg = Config.fromfile("configs/surface_guided/configE.py")
anonymizer = build_anonymizer(
    cfg, detection_score_threshold=0.3)

iface = gradio.Interface(
    anonymize, [gradio.inputs.Image(type="pil", label="Upload your image or try the example below!"), gradio.inputs.Slider(minimum=0, maximum=1, step=0.01, default=0.5, label="Truncation value (set to >0 to generate different bodies between runs)")],
    examples=[["coco_val2017_000000001000.jpg", None]],
    outputs="image",
    title="Realistic Full-Body Anonymization with Surface-Guided GANs",
    description="A live demo of full-body anonymization with surface guided GANs. See paper/code at: github.com/hukkelas/full_body_anonymization")
iface.launch()