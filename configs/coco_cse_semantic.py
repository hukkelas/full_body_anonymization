import os

imsize = (288, 160)
semantic_nc = 26
cse_nc = 16

dataset_type = "CocoCSESemantic"
data_root = os.environ["BASE_DATASET_PATH"] if "BASE_DATASET_PATH" in os.environ else "data"
data_root = os.path.join(data_root, "coco_cse")
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
        dict(type="Vertx2Semantic"),
        dict(type="CreateEmbedding"),
        dict(type="OneHotSemanticLabel", n_semantic=26),
        dict(type="Resize"),
        dict(type="Normalize", mean=(0.5,), std=(0.5,), inplace=True),
        dict(type="CreateCondition"),]
)
data_val = dict(
    dataset=dict(
        type=dataset_type,
        dirpath=os.path.join(data_root, "val"),
    ),
    cpu_transforms=[
    ],
    image_gpu_transforms=[
        dict(type="Vertx2Semantic"),
        dict(type="CreateEmbedding"),
        dict(type="OneHotSemanticLabel", n_semantic=26),
        dict(type="Resize"),
        dict(type="Normalize", mean=(0.5,), std=(0.5,), inplace=True),
        dict(type="CreateCondition"),
    ],
)

fid_real_directory = os.path.join(data_root, "val", "images")



semantic_labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    {"name": "background", "color": (  0,  0,  0)}, # 0
    {"name": "border", "color": (  170, 130, 180)}, # 1
    {"name": "right hand", "color": ( 70, 130, 180)}, # 2
    {"name": "right up leg", "color":(0, 0, 90)},  # 3
    {"name": "left arm", "color":(0, 0, 190)}, # 4 
    {"name": "left leg", "color":(70, 70, 70)}, # 5
    {"name": "left toe base", "color": (70, 70, 170)}, # 6
    {"name": "left foot", "color": (255, 0, 0)}, # 7
    {"name": "spine1", "color": (150, 0, 0)}, # 8
    {"name": "spine2", "color": (255, 0, 100)}, # 9
    {"name": "left shoulder", "color": (150, 0, 100)}, # 10
    {"name": "right shoulder", "color": (207, 142, 35)}, # 11
    {"name": "right foot", "color": (107, 142, 35)}, # 12
    {"name": "head", "color": (207, 142, 135)}, # 13
    {"name": "right arm", "color": (107, 142, 135)}, # 14
    {"name": "left hand index 1", "color": (255, 120, 60)}, # 15
    {"name": "right leg", "color": (180, 120, 60)}, # 16
    {"name": "right hand index 1", "color": (255, 120, 160)},  # 17
    {"name": "left fore arm", "color": (180, 120, 160)}, # 18
    {"name": "right fore arm", "color": (255, 20, 160)}, # 19
    {"name": "neck", "color": (180, 20, 160)}, # 20
    {"name": "right toe base", "color": (255, 20, 60)}, # 21
    {"name": "spine", "color": (180, 20, 60)}, # 22
    {"name": "left up leg", "color":  (0, 0, 142)}, # 23
    {"name": "left hand", "color": (0, 0, 70)}, # 24
    {"name": "hips", "color": (150, 0, 0)}, # 25
]
