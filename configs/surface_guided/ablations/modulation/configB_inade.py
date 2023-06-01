_base_config_ = ["configB_spade.py"]

generator = dict(
    style_cfg=dict(
        encoder_modulator="INADE",
        decoder_modulator="INADE",
        middle_modulator="INADE",

    ),
    embed_z=False,
    class_specific_z=True,
)
