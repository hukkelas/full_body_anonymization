# Realistic Full-Body Anonymization with Surface-Guided GANs

This is the official source code for the paper "Realistic Full-Body Anonymization with Surface-Guided GANs".

[[Arixv Paper]](https://arxiv.org/abs/2201.02193)
[[Appendix]](https://folk.ntnu.no/haakohu/fba_appendix.pdf)

Surface-guided GANs is an automatic full-body anonymization technique based on Generative Adversarial Networks.
![](docs/figures/architecture.jpg)

The key idea of surface-guided GANs is to guide the generative model with dense pixel-to-surface information (based on [continuous surface embeddings](https://arxiv.org/abs/2011.12438)). This yields highly realistic anonymization result and allows for diverse anonymization.
![](docs/figures/method.jpg)


## Requirements
- Pytorch >= 1.9
- Torchvision >= 0.11
- Python >= 3.8
- CUDA capable device for training. Training was done with 1-4 32GB V100 GPUs.

## Installation

We recommend to setup and install pytorch with [anaconda](https://www.anaconda.com/) following the [pytorch installation instructions](https://pytorch.org/get-started/locally/).

1. Clone repository: `git clone https://github.com/hukkelas/full_body_anonymization/`.
2. Install using `setup.py`:
```
pip install -e .
```


## Test the model

The file `anonymize.py` can anonymize image paths, directories and videos. `python anonymize.py --help` prints the different options.

To anonymize, visualize and save an output image, you can write:
```
python3 anonymize.py configs/surface_guided/configE.py coco_val2017_000000001000.jpg --visualize --save
```
The truncation value decides the "creativity" of the generator, which you can specify in the range (0, 1). Setting `-t 1` will generate diverse anonymization between individuals in the image.
We recommend to set it to `t=0.5` to tradeoff between quality and diversity.

```
python3 anonymize.py configs/surface_guided/configE.py coco_val2017_000000001000.jpg --visualize --save -t 1
```

## Pre-trained models
Current release includes a pre-trained model for ConfigE from the main paper.
More pre-trained models will be released later.

## Train the model
Instructions to train and reproduce results from the paper will be released by January 14th 2022.

## License
All code, except the stated below, is released under [MIT License](License).

Code under has are provided with other licenses:
- `torch_utils/`: Code modified from [github.com/NVlabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch). Separate license is attached in the directory.
- `dnnlib/`: Code modified from [github.com/NVlabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch). Separate license is attached in the directory.
- Detection network: See [Detectron2 License](https://github.com/facebookresearch/detectron2/blob/main/LICENSE).

## Citation
If you use this code for your research, please cite:
```
@misc{hukkelås2022realistic,
      title={Realistic Full-Body Anonymization with Surface-Guided GANs}, 
      author={Håkon Hukkelås and Morten Smebye and Rudolf Mester and Frank Lindseth},
      year={2022},
      eprint={2201.02193},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}