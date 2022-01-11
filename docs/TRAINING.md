# Training
You need 1 or more GPUs to train/evaluate.

### 1. Dataset setup
First, setup the dataset as described in [DATASETS.md](DATASETS.md).

### 2. Training
You need a GPU to train or evaluate.

You can start train with `train.py` and the desired config file:
```
python train.py configs/....
```
The script will start training automatically with all available CUDA devices. You can change the number of GPUs to train with `CUDA_VISIBLE_DEVICES`. For example, to only train on GPU 0 and 1:
```
CUDA_VISIBLE_DEVICES=0,1 python train.py configs/....
```
Outputs will be saved to `_output_dir`in [`configs/defaults.py`](../configs/defaults.py) (by default this is "outputs").
All metrics will be periodically logged (see [here](#logging-of-data) for setup instructions).


## Evaluation.
1. Main metrics:
```
python3 validate.py configs/...
```

2. To calculate invariance to affine transformations:
```
python3 -m tools.evaluation.affine_invariance configs/...
```
3. To calculate FID of the face region:
```
python3 -m tools.evaluation.face_fid configs/...
```
4. To calculate PPL:
```
python3 -m tools.evaluation.ppl configs/...
```


## FAQ


### How does the configuration files work?
The configuration files work by separating different configs into [defaults](../configs/defaults.py) (hyperparameters shared across datasets and different models) and dataset-specific configs (e.g. [coco_cse](../configs/coco_cse.py)).
To extend a config, you set the relative path for the config you want to extend in the variable `_base_config_`. `_base_config_` accepts both strings and lists, where configs are read iteratively and will override values that have the same key.


**Example config:**
```python
_base_config_ = ["configs/coco_cse.py", "configs/defaults.py"]
optimizer.D_opts.lr = 0.001*3 # This will overwrite the value in configs/defaults.py
```
This will read from the files from left-to-right and overwrite any values that have matching keys. 


### GPU vs CPU data transforms
The code splits transforms into GPU-specific and CPU-specific transforms, depending on which transforms performs the fastest on GPU vs CPU.
See [`configs/coco_cse.py`](../configs/coco_cse.py) for examples.

### Logging of data
The code supports logging with tensorboard, and wandb. You can change this by setting `logging_backend` in [`configs/defaults.py`](../configs/defaults.py) to the following values: "tensorboard", "wandb", "none".
Output directory of tensorboard training is outputted at the start of training.

Follow the [setup instructions](https://docs.wandb.ai/quickstart#1-set-up-wandb) to setup wandb.