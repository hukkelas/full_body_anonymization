_base_config_ = ["configA.py"]

checkpoint_url = "https://folk.ntnu.no/haakohu/checkpoints/surface_guided/configB.torch"


loss = dict(
    gan_criterion=dict(type="fpn_cse", l1_weight=1, lambda_real=1, lambda_fake=0)
)
generator = dict(
    input_cse=True
)