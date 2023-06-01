# Reproducing paper models

To calculate the metrics presented in the article, see [TRAINING.md](TRAINING.md).
All models are trained with 2x NVIDIA V100-32GB, except configE which is trained with 4 V100's.


The following configs were used for the tables in the paper

### Table 1
```
configs/surface_guided/configA.py
configs/surface_guided/configB.py
configs/surface_guided/configC.py
configs/surface_guided/configD.py
configs/surface_guided/configE.py
```
### Table 2
```

configs/surface_guided/ablations/f_depth/configD_n0.py
configs/surface_guided/ablations/f_depth/configD_n2.py
configs/surface_guided/ablations/f_depth/configD_n4.py
configs/surface_guided/configD.py
```

### Table 3
```
configs/surface_guided/configB.py
configs/surface_guided/ablations/modulation/configB_spade.py
configs/surface_guided/ablations/modulation/configB_inade.py
configs/surface_guided/ablations/modulation/configB_clade.py
configs/surface_guided/ablations/modulation/configB_stylegan.py
configs/surface_guided/ablations/modulation/configB_comod.py
configs/surface_guided/configC.py
configs/surface_guided/configD.py
```


#### Computing invariance to affine transformations
```
python -m tools.evaluation.affine_invariance configs/surface_guided/configD.py
```
#### Computing Face FID
```
python -m tools.evaluation.face_fid configs/surface_guided/configD.py
```

### Table 6 - COCO Anonymization results
The following describes how to generate anonymized versions of the COCO dataset.
To generate the final metrics, see the documentation of [detectron2](https://www.github.com/facebookresearch/detectron2).
We used [configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml](https://github.com/facebookresearch/detectron2/blob/335b19830e4ea5c5a74a085a04ff4a2f1a1dbf71/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml) and commit `335b19830e4ea5c5a74a085a04ff4a2f1a1dbf71`.

1. Detect on dataset first:
```
python3 -m tools.coco_anonymization.dump_detections configs/surface_guided/configE.py path/to/coco/dataset
```
2. Then, anonymize the dataset:
```
python3 -m tools.coco_anonymization.anonymize_dataset configs/surface_guided/configE.py path/to/coco/dataset path/to/coco/dataset_anonymized
```
This will save the anonymized dataset to `path/to/coco/dataset_anonymized`, which you then can use to train/validate with detectron2.

To reproduce results with different types of traditional anonymizers, replace `configE.py` with a config from [configs/anonymizers](../configs/anonymizers).