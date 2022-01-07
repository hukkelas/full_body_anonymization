_base_config_ = ["configA.py"]

checkpoint_url = "https://folk.ntnu.no/haakohu/checkpoints/surface_guided/configC.torch"

generator = dict(
    use_norm=True,
    style_cfg=dict(
        type="CSEStyleMapper", encoder_modulator="CSELinear", decoder_modulator="CSELinear",middle_modulator="CSELinear",
        w_mapper=dict(input_z=False)),
)
loss = dict(
    gan_criterion=dict(type="fpn_cse", l1_weight=1, lambda_real=1, lambda_fake=0)
)