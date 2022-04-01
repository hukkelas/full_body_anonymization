import zipfile
import numpy as np
from PIL import Image
import click
from pathlib import Path
import requests
import tqdm
import json

def download_file(url: str, path: Path):
    if path.is_file():
        print(f"File is already downloaded to: {path.absolute()}.\n\tDelete the file to redownload.")
        return
    response = requests.get(url, stream=True)
    total_length = int(response.headers.get("content-length"))
    assert response.status_code == 200, \
        f"Could not download file: {response.status_code}"
    path.parent.mkdir(exist_ok=True, parents=True)
    print(f"Downloading file from:\n\t{url}")
    with open(path, "wb") as fp:
        for data in tqdm.tqdm(
                response.iter_content(chunk_size=4096), total=total_length/4096,
                desc=f"Download."):
            fp.write(data)
    print(f"File downloaded to: {path}")


def check_coco_directory(coco_path: Path):
    assert coco_path.is_dir(), f"Did not find COCO directory in: {coco_path}"
    val_dir = coco_path.joinpath("val2014")
    assert val_dir.is_dir(), f"Did not find validation images in: {val_dir}"
    train_dir = coco_path.joinpath("train2014")
    assert train_dir.is_dir(), f"Did not find training images in: {train_dir}"
    print("COCO Dataset direcotry structure is OK.")


def download_metainfo(split: str, target_dir: Path):
    split2box_url = {
        "minival": "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/cdae70b8-f0ee-4396-82ec-97df09bc535d36a553ba-2680-4b07-afc6-da0299abea47507df925-02e1-47f2-986e-d5d13b552b3a",
        "val": "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/d1080095-9ff9-44ca-9dd5-dad18fb2e8a4ea9f8e0d-d0a2-4a44-900b-172c3a26c4613a0b8141-d729-4064-bf22-f88ded0a9007",
        "train": "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/edeed05e-ef29-4a6f-b6b0-42e61e490c38e708e9fc-71b3-471e-b6ff-1cdcfcd65d4b6f18b997-8ef7-4dbf-9f55-c09ea61f9eb3"
    }
    url = split2box_url[split]
    metainfo_path = target_dir.joinpath("metainfo.json")
    download_file(url, metainfo_path)

    with open(metainfo_path, "r") as fp:
        data = json.load(fp)
    return data

def download_and_extract_embedding(split:str, target_dir: Path):
    split2zip_url = {
        "minival": "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/036aaaea-031f-410c-9e8d-c5a02a322a246182d344-2ea3-49ce-8bf2-ef186fe763c6e27b9406-b57e-4e41-b355-e969da0ffc43",
        "val": "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/fa50ca16-7251-45ac-8945-884ccb93c251020a2b7d-b910-498e-b9ca-7dd4b7de99f13f719b86-44e7-4762-ad8a-f34ca453b1fd",
        "train": "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/cbc10537-637e-4020-a1e4-2dfb7b75dd51d889758b-6ee8-4f4f-95a9-241c6c1697046ec7c642-e074-466a-ba35-7a08894cc477"
    }
    url = split2zip_url[split]
    embedding_zip_path = target_dir.joinpath("embeddings.zip")
    download_file(url, embedding_zip_path)
    embedding_path = target_dir.joinpath("embedding")
    print("Extracing embeddings to:", embedding_path)
    with zipfile.ZipFile(embedding_zip_path, "r") as fp:
        fp.extractall(embedding_path)
    print(f"You can delete the zipfile: {embedding_zip_path} to clean up hard drive space.")


def generate_coco_cse(split: str, coco_path: Path, target_path: Path):

    target_dir = target_path.joinpath(split)
    print(f"Saving data {split} to {target_dir}")
    cse_imdir = target_dir.joinpath("images")
    cse_imdir.mkdir(exist_ok=True, parents=True)
    print(f"Saving images to: {cse_imdir}")

    metainfo = download_metainfo(split, target_dir)


    download_and_extract_embedding(split, target_dir)
    for coco_cse_image_id, info in tqdm.tqdm(metainfo.items(), desc="Generating images"):
        coco_image_name = info["image_name"]
        coco_impath = coco_path.joinpath(coco_image_name)
        if not coco_impath.is_file():
            raise FileNotFoundError("Did not find image in path: " + str(coco_impath.absolute()))
        coco_image = np.array(Image.open(coco_impath))
        l, t, r, b = info["box"]
        im = coco_image[t:b, l:r]
        target_impath = cse_imdir.joinpath(coco_cse_image_id + ".png")
        Image.fromarray(im).save(target_impath)
        
    for coco_cse_image_id in tqdm.tqdm(metainfo.keys(), desc="Validating correctness of file structure"):
        embedding_path = target_dir.joinpath("embedding", coco_cse_image_id + ".npy")
        assert embedding_path.is_file(), \
            f"Expected embedding to exist in {embedding_path}"
        impath = cse_imdir.joinpath(coco_cse_image_id + ".png")
        Image.open(impath).verify()
    embed_map_url = "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/af0097bf-408c-4853-88f1-678567bfaebbb0366e41-4d08-43af-a486-0ffa151f97cf4665c1d3-2821-47eb-820a-d0969ca677dc"
    embed_map_path = target_dir.joinpath("embed_map.npy")
    download_file(embed_map_url, embed_map_path)
    print("The dataset was successfully downloaded to:", target_dir)



@click.command()
@click.argument("coco_path")
@click.argument("target_path")
@click.argument("split", default="train", type=click.Choice(["train", "val", "minival", "all"]))
def main(coco_path: str, target_path: str, split):
    """
    Downloads and extracts the COCO-CSE dataset.
    """
    coco_path  = Path(coco_path)
    target_path = Path(target_path)
    if split == "all":
        splits = ["train", "val"," minival"]
    else:
        splits = [split]
    check_coco_directory(coco_path)

    for split in splits:
        generate_coco_cse(split, coco_path,  target_path)

    

if __name__ == "__main__":
    main()
