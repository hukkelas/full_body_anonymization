_base_config_ = ["../../configB.py"]

generator = dict(
    type="ComodGenerator",
    style_cfg=dict(type="CoModStyleMapper", only_gamma=True),
    embed_z=False,
    latent_space="w_comodgan"
)
