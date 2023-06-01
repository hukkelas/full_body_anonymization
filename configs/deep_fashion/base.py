_base_config_ = ["../deep_fashion_highres.py", "../defaults.py"]
project = "deep_fashion"

max_images_to_train = int(8e6)

optimizer = dict(D_opts=dict(lr=0.002), G_opts=dict(lr=0.002))

loss = dict(
    gan_criterion=dict(type="uncond_fpn_cse", l1_weight=1, lambda_real=1, lambda_fake=0),
    gradient_penalty=dict(mask_out=False),
)

generator = dict(
    type="DecoderGenerator",
    cnum=48,
    max_cnum_mul=16,
    z_channels=512,
    style_cfg=dict(
        type="UnconditionalCSEStyleMapper",
        w_mapper=dict(input_z=True, n_layer_z=2),
        only_gamma=True),
    embed_z=False,
    class_specific_z=False,
    conv_clamp=256,
)

discriminator=dict(
    type="FPNDiscriminator",
    output_fpn=True,
    cnum=32,
    max_cnum_mul=16,
    input_condition=False,
)
