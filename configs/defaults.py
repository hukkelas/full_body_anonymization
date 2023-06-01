import os
imsize = (288, 160)
semantic_nc = None
image_channels = 3
max_images_to_train = 12e6
cse_nc = None
project = "fba"
semantic_labels = None

_output_dir = os.environ["BASE_OUTPUT_DIR"] if "BASE_OUTPUT_DIR" in os.environ else "outputs"
_cache_dir = ".fba_cache"
_checkpoint_dir = "checkpoints"
logger_backend = "wandb" # choices: ["tensorboard", "none"]

# Tag used for matplotlib plots
log_tag = None

# URL for publishing models
checkpoint_url = None
metrics_url = None

optimizer = dict(
    lazy_regularization=True,
    lazy_reg_interval=16,
    D_opts=dict(type="Adam", lr=0.001, betas=(0.0, 0.99)),
    G_opts=dict(type="Adam", lr=0.001, betas=(0.0, 0.99)),
)

# exponential moving average
EMA = dict(nimg_half_time=10e3, rampup_nimg=0)

hooks = {
    "time": dict(type="TimeLoggerHook", num_ims_per_log=10e3),
    "metric": dict(type="MetricHook", ims_per_log=2e5, n_diversity_samples=1),
    "checkpoint": dict(type="CheckpointHook", ims_per_checkpoint=2e5, test_checkpoints=[]),
    "image_saver": dict(type="ImageSaveHook", ims_per_save=2e5, n_diverse_samples=4, n_diverse_images=8, nims2log=16, save_train_G=False),
}

ims_per_log = 2048
random_seed = 0

jit_transform = False
data_train = dict(
    loader=dict(num_workers=8, drop_last=True, pin_memory=True, batch_size=32, prefetch_factor=2),
    sampler=dict(drop_last=True, shuffle=True)
)

data_val = dict(
    loader=dict(num_workers=8, pin_memory=True, batch_size=32, prefetch_factor=2),
    sampler=dict(drop_last=True, shuffle=False)
)

loss = dict(
    type="LossHandler",
    gan_criterion=dict(type="nsgan", weight=1),
    gradient_penalty=dict(type="r1_regularization", weight=5, mask_out=True),
    epsilon_penalty=dict(type="epsilon_penalty", weight=0.001),
)

generator = dict(
    type="UnetGenerator",
    scale_grad=True,
    min_fmap_resolution=32,
    cnum=32,
    max_cnum_mul=16,
    n_middle_blocks=2,
    z_channels=512,
    mask_output=True,
    input_semantic=False,
    style_cfg=dict(type="NoneStyle"),
    embed_z=True,
    class_specific_z=False,
    conv_clamp=256,
    input_cse=False,
    latent_space=None,
    use_cse=True,
    modulate_encoder=False,
    norm_type="instance_norm_std",
    norm_unet=False,
    unet_skip="residual"
)


discriminator = dict(
    type="FPNDiscriminator",
    min_fmap_resolution=8,
    max_cnum_mul=8,
    cnum=32,
    input_condition=True,
    semantic_input_mode=None,
    conv_clamp=256,
    input_cse=False,
    output_fpn=False,
)
