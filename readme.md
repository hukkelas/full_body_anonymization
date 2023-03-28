# Realistic Full-Body Anonymization with Surface-Guided GANs

This is the official source code for the paper "Realistic Full-Body Anonymization with Surface-Guided GANs".

[[Arixv Paper]](https://arxiv.org/abs/2201.02193)
[[Appendix]](https://folk.ntnu.no/haakohu/fba_appendix.pdf)
[[Google Colab demo]](https://colab.research.google.com/drive/10bxR6AOityusLFiTKT9ZUoJ5wMDkvCfe?usp=sharing)
[[WACV 2023 Conference Presentation]](https://youtu.be/ttcE-u-pDxk)

Surface-guided GANs is an automatic full-body anonymization technique based on Generative Adversarial Networks.
![](docs/figures/architecture.jpg)

The key idea of surface-guided GANs is to guide the generative model with dense pixel-to-surface information (based on [continuous surface embeddings](https://arxiv.org/abs/2011.12438)). This yields highly realistic anonymization result and allows for diverse anonymization.
![](docs/figures/method.jpg)


### Check out the new [DeepPrivacy2](https://github.com/hukkelas/deep_privacy2)! It significantly improves anonymization quality compared to this repository.

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

Otherwise, you can setup your environment with our provided [Dockerfile](Dockerfile).


## Test the model


### Anonymizing files
The file `anonymize.py` can anonymize image paths, directories and videos. `python anonymize.py --help` prints the different options.

To anonymize, visualize and save an output image, you can write:
```
python3 anonymize.py configs/surface_guided/configE.py coco_val2017_000000001000.jpg --visualize --save
```
The truncation value decides the "creativity" of the generator, which you can specify in the range (0, 1). Setting `-t 1` will generate diverse anonymization between runs.
For config A/B/C, the truncation value accepts range of (0, $\infty$). Setting `-t=None` will apply to latent truncation.
```
python3 anonymize.py configs/surface_guided/configE.py coco_val2017_000000001000.jpg --visualize --save -t 1
```
### Gradio App
Check out the interactive demo with our [gradio implementation](app.py).
Run
```
python3 app.py
```

## Train the model
See [docs/TRAINING.md](docs/TRAINING.md).

## Reproducing paper results
See [docs/REPRODUCING.md](docs/REPRODUCING.md).


## License
All code, except the stated below, is released under [MIT License](License).

Code under:
- `torch_utils/`: Code modified from [github.com/NVlabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch). Separate license is attached in the directory.
- `dnnlib/`: Code modified from [github.com/NVlabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch). Separate license is attached in the directory.
- Detection network: See [Detectron2 License](https://github.com/facebookresearch/detectron2/blob/main/LICENSE).

## Citation
If you use this code for your research, please cite:
```
@inproceedings{hukkelas23FBA,
  author={Hukkelås, Håkon and Smebye, Morten and Mester, Rudolf and Lindseth, Frank},
  booktitle={2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)}, 
  title={Realistic Full-Body Anonymization with Surface-Guided GANs}, 
  year={2023},
  volume={},
  number={},
  pages={1430-1440},
  doi={10.1109/WACV56688.2023.00148}}

