_base_config_ = ["base.py"]
generator = dict(
    use_norm=True,
    style_cfg=dict(
        type="SemanticStyleEncoder",
        encoder_modulator="SPADE",
        decoder_modulator="SPADE",middle_modulator="SPADE",
        nhidden=256),
    use_cse=False
)
discriminator=dict(
    pred_only_cse=False,
    pred_only_semantic=True
)
loss = dict(
    gan_criterion=dict(type="segmentation", seg_weight=.1)
)
