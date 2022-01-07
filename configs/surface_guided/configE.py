_base_config_ = ["configA.py"]
max_images_to_train = 10e6

checkpoint_url = "https://folk.ntnu.no/haakohu/checkpoints/surface_guided/configE.torch"

generator = dict(
    use_norm=True,
    cnum=64,
    style_cfg=dict(
        type="CSEStyleMapper", encoder_modulator="CSELinear", decoder_modulator="CSELinear",middle_modulator="CSELinear",
        w_mapper=dict(input_z=True)),
    embed_z=False
)
loss = dict(
    gan_criterion=dict(type="fpn_cse", l1_weight=.1, lambda_real=1, lambda_fake=0)
)

discriminator=dict(
        cnum=64)

optimizer=dict(D_opts=dict(lr=0.002), G_opts=dict(lr=0.002))

