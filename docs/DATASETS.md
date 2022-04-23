# Dataset


## Downloading the COCO CSE dataset

The script `tools/download_coco_cse.py` will automatically download the COCO-CSE dataset.
To download the dataset, do the following:

1. Download the train2014 and val2014 images from the COCO dataset.
The script expects the following directory file structure:
```
PATH_TO_COCO
    - val2014
        - *.jpg
    - train2014
        - *.jpg
```

You can download it with the following:
```bash
mkdir -p data/coco # save images under the subfolder data/coco
cd data/coco
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
unzip train2014.zip
unzip val2014.zip
cd ../..
```

2. Run the `download_coco_cse.py` script. The script has three arguments, path to the coco dataset (COCO_PATH), save path for coco_cse (TARGET_PATH) and the split to download.
```
python tools/download_coco_cse.py PATH_TO_COCO PATH_TO_COCO_CSE DATASET_SPLIT_TO_DOWNLOAD
```

To download all subsets, use the following command:
```
python3 tools/download_coco_cse.py data/coco data/coco_cse all
```
To download only minival:
```
python3 tools/download_coco_cse.py data/coco data/coco_cse minival
```
To download only validation:
```
python3 tools/download_coco_cse.py data/coco data/coco_cse val
```
To download only train:
```
python3 tools/download_coco_cse.py data/coco data/coco_cse train
```

The script will generate the following structure
```bash
PATH_TO_COCO_CSE
    - train
        - images/
            - *.png
        - embedding
            - *.npy # embedding for each image
        - embed_map.npy # the 16-channel CSE embedding for each vertex
    - train
        - images/
            - *.png
        - embedding
            - *.npy # embedding for each image
        - embed_map.npy # the 16-channel CSE embedding for each vertex
    - train
        - images/
            - *.png
        - embedding
            - *.npy # embedding for each image
        - embed_map.npy # the 16-channel CSE embedding for each vertex
```
embeddings can be loaded with
```python
vertices, mask, border = np.split(np.load("embedding_path.npy"), 3, axis=-1)
```
where vertices are indices to embed_map.npy, mask is the dilated mask (1 for known values, 0 for missing), and border is the dilated region (0 for non-dilated regions).
To get the mask where the vertices are known (ie. corresponds to the body), you can find it with:
```
E_mask = 1 - (mask + border)
```
where E_mask will be 1 for all pixels that has a CSE embedding.
