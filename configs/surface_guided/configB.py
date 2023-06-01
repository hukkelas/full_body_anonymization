_base_config_ = ["configA.py"]

generator=dict(
    input_cse=True,
)

loss = dict(
    gan_criterion=dict(type="fpn_cse", l1_weight=.1, lambda_real=1, lambda_fake=0)
)
discriminator=dict(
    output_fpn=True,
)
