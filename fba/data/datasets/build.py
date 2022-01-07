from fba.utils import Registry, build_from_cfg

DATASET_REGISTRY = Registry("DATASET")


def build_dataset(data_cfg, imsize, transform, is_train):
    additional_kwargs = {"transform": transform}
    if data_cfg.type not in ["Cityscapes"]:
        additional_kwargs.update({
            "imsize": imsize, "is_train": is_train
        })
    return build_from_cfg(
        data_cfg,
        DATASET_REGISTRY,
        **additional_kwargs
    )
