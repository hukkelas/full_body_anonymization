import os


imsize = (288, 160)
semantic_nc = None
cse_nc = 16

dataset_type = "CocoCSE"
data_root = os.path.join("data", "coco_cse")

data_train = dict(
    dataset=dict(
        type=dataset_type,
        dirpath=os.path.join(data_root, "train"),
    ),
    cpu_transforms=[
    ],
    image_gpu_transforms=[
        # If not enough augmentation, can set higher for most values...
        dict(type="StyleGANAugmentPipe",
            rotate=0.5, rotate_max=.05,
            xint=.5, xint_max=0.05,
            scale=.5, scale_std=.05,
            aniso=0.5, aniso_std=.05,
            xfrac=.5, xfrac_std=.05,
            brightness=.5, brightness_std=.05,
            contrast=.5, contrast_std=.1,
            hue=.5, hue_max=.05,
            saturation=.5, saturation_std=.5,
            imgfilter=.5, imgfilter_std=.1),
        dict(type="RandomHorizontalFlip", p=0.5),
        dict(type="CreateEmbedding"),
        dict(type="Resize"),
        dict(type="Normalize", mean=(0.5,), std=(0.5,), inplace=True),
        dict(type="CreateCondition"),
    ],
)
data_val = dict(
    dataset=dict(
        type=dataset_type,
        dirpath=os.path.join(data_root, "val"),
    ),
    cpu_transforms=[
    ],
    image_gpu_transforms=[
        dict(type="CreateEmbedding"),
        dict(type="Resize"),
        dict(type="Normalize", mean=(0.5,), std=(0.5,), inplace=True),
        dict(type="CreateCondition"),
    ],
)

fid_real_directory = os.path.join(data_root, "val", "images")
