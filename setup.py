import torch
import torchvision
from setuptools import setup, find_packages

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 9], "Requires PyTorch >= 1.9"
torchvision_ver = [int(x) for x in torchvision.__version__.split(".")[:2]]
assert torchvision_ver >= [0, 11], "Requires torchvision >= 0.11"


setup(
    name="fba",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "cython",
        "matplotlib",
        "tqdm",
        "tflib",
        "autopep8",
        "tensorboard",
        "opencv-python",
        "requests",
        "pyyaml",
        "addict",
        "scikit-image",
        #"detectron2 @ git+https://github.com/facebookresearch/detectron2@main#egg=detectron2",
        "detectron2-densepose @ git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose",
        "kornia",
        "torch_fidelity",
        "ninja",
        "moviepy"
    ],
)
