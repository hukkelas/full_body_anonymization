_base_config_ = ["../../../coco_cse_semantic.py", "../../../defaults.py"]

generator = dict(
    style_cfg=dict(
        type="SemanticStyleEncoder", only_gamma=True,
        encoder_modulator="SPADE",
        decoder_modulator="SPADE",
        middle_modulator="SPADE",
        nhidden=256
    ),
    embed_z=True,
    input_cse=False,
    input_semantic=True,
    use_cse=False
)
loss = dict(
    gan_criterion=dict(type="fpn_cse", l1_weight=.1, lambda_real=1, lambda_fake=0)
)
discriminator=dict(
    output_fpn=True,
)
