_base_config_ = ["../coco_cse_semantic.py", "../defaults.py"]
generator = dict(
    use_cse=False
)

loss = dict(
    gan_criterion=dict(type="fpn_cse", l1_weight=1, lambda_real=1, lambda_fake=0),
)
