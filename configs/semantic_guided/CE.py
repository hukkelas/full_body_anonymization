_base_config_ = ["base.py"]

generator = dict(
    input_cse=True,
    use_cse=True
)
discriminator=dict(
    pred_only_cse=False,
    pred_only_semantic=True
)

loss = dict(
    gan_criterion=dict(type="segmentation", seg_weight=.1)
)
