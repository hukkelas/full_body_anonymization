import os


imsize = (384, 256)
semantic_nc = None
cse_nc = 16

dataset_type = "DeepFashion"
data_root = os.environ["BASE_DATASET_PATH"] if "BASE_DATASET_PATH" in os.environ else "data"
data_root = os.path.join(data_root, "deep_fashion_highres")

data_train = dict(
    dataset=dict(
        type=dataset_type,
        dirpath=data_root,
        subset_split_file="train_images.txt"
    ),
    cpu_transforms=[
    ],
    image_gpu_transforms=[
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
        dirpath=data_root,
        subset_split_file="test_images.txt"
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
