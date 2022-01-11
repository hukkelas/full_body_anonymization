_base_config_ = ["../deep_fashion_highres.py", "../defaults.py"]
project = "deep_fashion"

max_images_to_train = int(8e6)

optimizer = dict(D_opts=dict(lr=0.002), G_opts=dict(lr=0.002))

loss = dict(
    gradient_penalty=dict(mask_out=False),
)

generator = dict(
    type="DecoderGenerator",
    min_fmap_resolution=32,
    cnum=48,
    max_cnum_mul=16,
    n_middle_blocks=2,
    z_channels=512,
    use_norm=True,
    style_cfg=dict(
        type="UnconditionalCSEStyleMapper", decoder_modulator="CSELinearSimple",
        w_mapper=dict(input_z=True)),
    embed_z=True,
    class_specific_z=False,
    conv_clamp=256,
)


discriminator = dict(
    max_cnum_mul=16,
    cnum=32,
    input_condition=False,
    semantic_input_mode=None,
    conv_clamp=256,
    input_cse=False
)