import os
import click
import torch
from fba import utils, logger
from fba.config import Config
import torch_fidelity
from fba.data import build_dataloader_val
from fba.infer import build_trained_generator
from fba.metrics.torch_metrics import ImageIteratorWrapper, lpips
import torch.nn.functional as F
import tqdm

@torch.no_grad()
def calculate_face_metrics(G, dataloader):
    images_real = []
    images_fake = []
    avg_lpips = 0
    for batch in tqdm.tqdm(iter(dataloader)):
        fake = G(**batch)["img"]
        real = batch["img"]
        for i in range(fake.shape[0]):
            bbox = batch["boxes_XYXY"][i].long().clamp(0)

            assert bbox.shape == (4,)
            if bbox[2] - bbox[0] < 8 or bbox[3] - bbox[1] < 8:
                continue
            x0, y0, x1, y1 = bbox
            real_ = utils.denormalize_img(F.interpolate(real[i:i+1, :, y0:y1, x0:x1], size=(299, 299), mode="bilinear", align_corners=False))
            fake_ = utils.denormalize_img(F.interpolate(fake[i:i+1, :, y0:y1, x0:x1], size=(299,299), mode="bilinear", align_corners=False))
            lpips_ = lpips(real_, fake_)
            avg_lpips += lpips_
            images_real.append((real_*255).byte().cpu())
            images_fake.append((fake_*255).byte().cpu())
    avg_lpips /= len(images_fake)
    print("LPIPS:", avg_lpips)
    print("STARTING FID CALCULATIONS!")
    fid_metrics = torch_fidelity.calculate_metrics(
            input1=ImageIteratorWrapper(images_real),
            input2=ImageIteratorWrapper(images_fake),
            cuda=torch.cuda.is_available(),
            fid=True,
            prc=True,
            batch_size=1,
            input1_model_num_samples=len(images_fake),
            input2_model_num_samples=len(images_real),
        )
    print(fid_metrics)
    to_return = {"metrics/face_box/LPIPS": avg_lpips}
    to_return.update({f"metrics/face_box/{k}": v for k, v in fid_metrics.items()})
    return to_return



@click.command()
@click.argument("config_path")
def main(config_path):
    dataset2FaceDataset = {
        "CocoCSE": "CocoCSEWithFace",
        "CocoCSESemantic": "CocoCSESemanticWithFace"
    }
    cfg = Config.fromfile(config_path)
    cfg.data_val.dataset.type = dataset2FaceDataset[cfg.data_val.dataset.type]
    G, global_step = build_trained_generator(cfg)
    cfg.data_val.loader.batch_size = 2
    dl = build_dataloader_val(cfg)
    face_metrics = calculate_face_metrics(G, dl)
    logger.init(cfg, resume=True)
    logger.update_global_step(global_step)
    logger.log_dictionary(face_metrics)
    logger.finish()



if __name__ == "__main__":
    if os.environ.get("AMP_ENABLED") is None:
        print("AMP not set. setting to True")
        os.environ["AMP_ENABLED"] = "1"
    else:
        assert os.environ["AMP_ENABLED"] in ["0", "1"]
    main()
