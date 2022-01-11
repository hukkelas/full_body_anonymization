_base_config_ = ["base.py"]
generator = dict(
    semantic_input_mode=None,
    use_norm=True,
    style_cfg=dict(
        type="CSEStyleMapper", encoder_modulator="CSELinear", decoder_modulator="CSELinear",middle_modulator="CSELinear",
        w_mapper=dict(input_z=True)),
    embed_z=False,
    use_cse=True
)
loss = dict(
    gan_criterion=dict(type="segmentation", seg_weight=.1)
)

discriminator=dict(
    pred_only_cse=False,
    pred_only_semantic=True
)