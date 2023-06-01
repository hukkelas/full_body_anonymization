_base_config_ = ["../../../coco_cse.py", "../../../defaults.py"]

generator = dict(
    style_cfg=dict(type="StyleGANMapper", only_gamma=True),
    embed_z=False,
    input_cse=True,
    latent_space="w_stylegan2"
)
loss = dict(
    gan_criterion=dict(type="fpn_cse", l1_weight=.1, lambda_real=1, lambda_fake=0)
)
discriminator=dict(
    output_fpn=True,
)
