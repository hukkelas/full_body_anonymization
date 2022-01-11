_base_config_ = ["base.py"]

generator = dict(
    semantic_input_mode="at_input"
)
discriminator=dict(pred_only_cse=True)
loss = dict(
    gan_criterion=dict(type="fpn_cse", l1_weight=1, lambda_real=1, lambda_fake=0)
)
