_base_config_ = ["base.py"]
generator = dict(
    use_norm=True,
    style_cfg=dict(
        type="SemanticStyleEncoder",
        encoder_modulator="CLADE",
        decoder_modulator="CLADE",middle_modulator="CLADE"),
    use_cse=False
)
discriminator=dict(pred_only_cse=True)
loss = dict(
    gan_criterion=dict(type="fpn_cse", l1_weight=1, lambda_real=1, lambda_fake=0)
)
