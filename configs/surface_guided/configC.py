_base_config_ = ["configB.py"]

generator = dict(
    style_cfg=dict(
        type="CSEStyleMapper", encoder_modulator="CSELinear", decoder_modulator="CSELinear",middle_modulator="CSELinear",
        w_mapper=dict(input_z=False)),
)