from fba.metrics.torch_metrics import psnr
from fba.data.transforms.functional import hflip, one_hot_semantic, vertx2semantic
from fba.infer import build_trained_generator
from fba.data import build_dataloader_val
import torch
import click
import numpy as np
from fba.config import Config
from fba import utils, logger
import scipy.signal
from torch_utils import misc
from torch_utils.ops import upfirdn2d
from torch_utils.ops import grid_sample_gradfix
from fba.data.transforms.stylegan2_transform import translate2d, scale2d, translate2d_inv, scale2d_inv, rotate2d_inv, wavelets, matrix


class StyleGANAugmentPipe(torch.nn.Module):
    def __init__(self,
        xflip=0, rotate90=0, xint=0, xint_max=0.125,
        scale=0, rotate=0, aniso=0, xfrac=0, scale_std=0.2, rotate_max=1, aniso_std=0.2, xfrac_std=0.125,
        ):
        super().__init__()
        self.register_buffer('p', torch.ones([]))       # Overall multiplier for augmentation probability.

        # Pixel blitting.
        self.rotate90         = float(rotate90)         # Probability multiplier for 90 degree rotations.
        self.xint             = float(xint)             # Probability multiplier for integer translation.
        self.xint_max         = float(xint_max)         # Range of integer translation, relative to image dimensions.

        # General geometric transformations.
        self.scale            = float(scale)            # Probability multiplier for isotropic scaling.
        self.rotate           = float(rotate)           # Probability multiplier for arbitrary rotation.
        self.aniso            = float(aniso)            # Probability multiplier for anisotropic scaling.
        self.xfrac            = float(xfrac)            # Probability multiplier for fractional translation.
        self.scale_std        = float(scale_std)        # Log2 standard deviation of isotropic scaling.
        self.rotate_max       = float(rotate_max)       # Range of arbitrary rotation, 1 = full circle.
        self.xflip = xflip


        # Setup orthogonal lowpass filter for geometric augmentations.
        self.register_buffer('Hz_geom', upfirdn2d.setup_filter(wavelets['sym6']))

        # Construct filter bank for image-space filtering.
        Hz_lo = np.asarray(wavelets['sym2'])            # H(z)
        Hz_hi = Hz_lo * ((-1) ** np.arange(Hz_lo.size)) # H(-z)
        Hz_lo2 = np.convolve(Hz_lo, Hz_lo[::-1]) / 2    # H(z) * H(z^-1) / 2
        Hz_hi2 = np.convolve(Hz_hi, Hz_hi[::-1]) / 2    # H(-z) * H(-z^-1) / 2
        Hz_fbank = np.eye(4, 1)                         # Bandpass(H(z), b_i)
        for i in range(1, Hz_fbank.shape[0]):
            Hz_fbank = np.dstack([Hz_fbank, np.zeros_like(Hz_fbank)]).reshape(Hz_fbank.shape[0], -1)[:, :-1]
            Hz_fbank = scipy.signal.convolve(Hz_fbank, [Hz_lo2])
            Hz_fbank[i, (Hz_fbank.shape[1] - Hz_hi2.size) // 2 : (Hz_fbank.shape[1] + Hz_hi2.size) // 2] += Hz_hi2
        self.register_buffer('Hz_fbank', torch.as_tensor(Hz_fbank, dtype=torch.float32))

    def get_G(self, batch):
        # Initialize inverse homogeneous 2D transform: G_inv @ pixel_out ==> pixel_in
        batch_size, num_channels, height, width = batch["img"].shape
        device = batch["img"].device
        I_3 = torch.eye(3, device=device)
        G_inv = I_3
        # Apply integer translation with probability (xint * strength).
        if self.xint > 0:
            t = (torch.rand([batch_size, 2], device=device) * 2 - 1) * self.xint_max
            t = torch.where(torch.rand([batch_size, 1], device=device) < self.xint * self.p, t, torch.zeros_like(t))
            G_inv = G_inv @ translate2d_inv(torch.round(t[:,0] * width), torch.round(t[:,1] * height))

        # --------------------------------------------------------
        # Select parameters for general geometric transformations.
        # --------------------------------------------------------

        # Apply isotropic scaling with probability (scale * strength).
        if self.scale > 0:
            s = torch.exp2(torch.randn([batch_size], device=device) * self.scale_std)
            s = torch.where(torch.rand([batch_size], device=device) < self.scale * self.p, s, torch.ones_like(s))
            G_inv = G_inv @ scale2d_inv(s, s)

        # Apply pre-rotation with probability p_rot.
        p_rot = 1 - torch.sqrt((1 - self.rotate * self.p).clamp(0, 1)) # P(pre OR post) = p
        if self.rotate > 0:
            theta = (torch.rand([batch_size], device=device) * 2 - 1) * np.pi * self.rotate_max
            theta = torch.where(torch.rand([batch_size], device=device) < p_rot, theta, torch.zeros_like(theta))
            G_inv = G_inv @ rotate2d_inv(-theta) # Before anisotropic scaling.

        # Apply anisotropic scaling with probability (aniso * strength).
        if self.aniso > 0:
            s = torch.exp2(torch.randn([batch_size], device=device) * self.aniso_std)
            s = torch.where(torch.rand([batch_size], device=device) < self.aniso * self.p, s, torch.ones_like(s))
            G_inv = G_inv @ scale2d_inv(s, 1 / s)

        # Apply post-rotation with probability p_rot.
        if self.rotate > 0:
            theta = (torch.rand([batch_size], device=device) * 2 - 1) * np.pi * self.rotate_max
            theta = torch.where(torch.rand([batch_size], device=device) < p_rot, theta, torch.zeros_like(theta))
            G_inv = G_inv @ rotate2d_inv(-theta) # After anisotropic scaling.

        # Apply fractional translation with probability (xfrac * strength).
        if self.xfrac > 0:
            t = torch.randn([batch_size, 2], device=device) * self.xfrac_std
            t = torch.where(torch.rand([batch_size, 1], device=device) < self.xfrac * self.p, t, torch.zeros_like(t))
            G_inv = G_inv @ translate2d_inv(t[:,0] * width, t[:,1] * height)

        do_hflip = np.random.random() < self.xflip
        return G_inv, I_3, do_hflip

    def forward(self, batch, G_inv, I_3, do_hflip):
        batch = {k: v.clone() for k,v in batch.items()}
        images = batch["img"]
        batch["vertices"] = batch["vertices"].float()
        assert isinstance(images, torch.Tensor) and images.ndim == 4
        batch_size, num_channels, height, width = images.shape
        device = images.device
        self.Hz_fbank = self.Hz_fbank.to(device)
        self.Hz_geom = self.Hz_geom.to(device)

        # Execute if the transform is not identity.
        if G_inv is not I_3:
            # Calculate padding.
            cx = (width - 1) / 2
            cy = (height - 1) / 2
            cp = matrix([-cx, -cy, 1], [cx, -cy, 1], [cx, cy, 1], [-cx, cy, 1], device=device) # [idx, xyz]
            cp = G_inv @ cp.t() # [batch, xyz, idx]
            Hz_pad = self.Hz_geom.shape[0] // 4
            margin = cp[:, :2, :].permute(1, 0, 2).flatten(1) # [xy, batch * idx]
            margin = torch.cat([-margin, margin]).max(dim=1).values # [x0, y0, x1, y1]
            margin = margin + misc.constant([Hz_pad * 2 - cx, Hz_pad * 2 - cy] * 2, device=device)
            margin = margin.max(misc.constant([0, 0] * 2, device=device))
            margin = margin.min(misc.constant([width-1, height-1] * 2, device=device))
            mx0, my0, mx1, my1 = margin.ceil().to(torch.int32)

            # Pad image and adjust origin.
            images = torch.nn.functional.pad(input=images, pad=[mx0,mx1,my0,my1], mode='reflect')
            batch["mask"] = torch.nn.functional.pad(input=batch["mask"], pad=[mx0,mx1,my0,my1], mode='constant', value=1.0)
            batch["border"] = torch.nn.functional.pad(input=batch["border"], pad=[mx0,mx1,my0,my1], mode='constant', value=0.0)
            batch["vertices"] = torch.nn.functional.pad(input=batch["vertices"], pad=[mx0,mx1,my0,my1], mode='constant', value=0.0)
            G_inv = translate2d((mx0 - mx1) / 2, (my0 - my1) / 2) @ G_inv

            # Upsample.
            images = upfirdn2d.upsample2d(x=images, f=self.Hz_geom, up=2)
            batch["mask"] = torch.nn.functional.interpolate(batch["mask"], scale_factor=2, mode="nearest")
            batch["border"] = torch.nn.functional.interpolate(batch["border"], scale_factor=2, mode="nearest")
            batch["vertices"] = torch.nn.functional.interpolate(batch["vertices"], scale_factor=2, mode="nearest")
            G_inv = scale2d(2, 2, device=device) @ G_inv @ scale2d_inv(2, 2, device=device)
            G_inv = translate2d(-0.5, -0.5, device=device) @ G_inv @ translate2d_inv(-0.5, -0.5, device=device)

            # Execute transformation.
            shape = [batch_size, num_channels, (height + Hz_pad * 2) * 2, (width + Hz_pad * 2) * 2]
            G_inv = scale2d(2 / images.shape[3], 2 / images.shape[2], device=device) @ G_inv @ scale2d_inv(2 / shape[3], 2 / shape[2], device=device)
            grid = torch.nn.functional.affine_grid(theta=G_inv[:,:2,:], size=shape, align_corners=False)
            images = grid_sample_gradfix.grid_sample(images, grid)

            batch["mask"] = torch.nn.functional.grid_sample(
                input=batch["mask"], grid=grid, mode='nearest', padding_mode="border", align_corners=False)
            batch["border"] = torch.nn.functional.grid_sample(
                input=batch["border"], grid=grid, mode='nearest', padding_mode="border", align_corners=False)
            batch["vertices"] = torch.nn.functional.grid_sample(
                input=batch["vertices"], grid=grid, mode='nearest', padding_mode="border", align_corners=False)

            # Downsample and crop.
            images = upfirdn2d.downsample2d(x=images, f=self.Hz_geom, down=2, padding=-Hz_pad*2, flip_filter=True)
            batch["mask"] = torch.nn.functional.interpolate(batch["mask"][:, :, Hz_pad*2:-Hz_pad*2, Hz_pad*2:-Hz_pad*2], scale_factor=.5, mode="nearest", recompute_scale_factor=False)
            batch["border"] = torch.nn.functional.interpolate(batch["border"][:, :, Hz_pad*2:-Hz_pad*2, Hz_pad*2:-Hz_pad*2], scale_factor=.5, mode="nearest", recompute_scale_factor=False)
            batch["vertices"] = torch.nn.functional.interpolate(batch["vertices"][:, :, Hz_pad*2:-Hz_pad*2, Hz_pad*2:-Hz_pad*2], scale_factor=.5, mode="nearest", recompute_scale_factor=False)
        batch["img"] = images
        batch["vertices"] = batch["vertices"]
        if do_hflip:
            base_transform = set(["landmarks", "img", "mask", "border", "vertices", "E_mask", "embed_map", "condition", "embedding", "vertx2cat"])
            batch1 = hflip({k: v for k, v in batch.items() if k in base_transform})
            if "semantic_mask" in batch:
                batch1.update({"semantic_mask": batch["semantic_mask"]})
            batch = batch1
        if "semantic_mask" in batch:
            n_semantic = batch["semantic_mask"].shape[1]
            batch = vertx2semantic(batch)
            batch = one_hot_semantic(batch, n_semantic)
        batch["condition"] = batch["img"] * batch["mask"]
        assert (batch["mask"][batch["border"] == 1] == 0).all()
        batch["E_mask"] = 1 - batch["mask"] - batch["border"]
        batch["embedding"] = batch["embed_map"][batch["vertices"].long().squeeze(dim=1)].permute(0, 3, 1, 2) * batch["E_mask"]
        return batch

@torch.no_grad()
def calculate_affine_invariance(dl, G, transform):
    n_images = 0
    total_psnr = 0
    for i, batch in enumerate(utils.tqdm_(dl)):
        G_inv, I_3, do_xflip = transform.get_G(batch)
        batch_rot = transform(batch, G_inv, I_3, do_xflip)
        utils.set_seed(i)
        img = G(**batch)["img"]
        img = transform({"img": img, "mask": batch["mask"], "border": batch["border"], "vertices":batch["vertices"], "embed_map": batch["embed_map"]}, G_inv, I_3, do_xflip)["img"]
        img = utils.denormalize_img(img)
        utils.set_seed(i)
        img_rot = G(**batch_rot)["img"]
        img_rot = utils.denormalize_img(img_rot)
        psnr_ = psnr(img.contiguous(), img_rot.contiguous())

        n_images += batch["img"].shape[0]
        total_psnr += psnr_.sum().item()
    print(total_psnr / n_images)
    return total_psnr / n_images


@click.command()
@click.argument("config_path")
@click.option("-n", "--num_images", default=8, type=int)
@torch.no_grad()
def main(config_path: str, num_images: int):
    utils.set_seed(0)
    cfg = Config.fromfile(config_path)
    G, global_step = build_trained_generator(cfg)

    cfg.data_val.loader.batch_size = num_images
    dl = build_dataloader_val(cfg)
    transform_translate = StyleGANAugmentPipe(
        xflip=0.0,
        xint=1, xint_max=0.125,
        scale=0, scale_std=0.2,
        rotate90=0,
        rotate=0, rotate_max=0.25,
        aniso=0, aniso_std=0.2,
        xfrac=0, xfrac_std=0.125
    )
    metrics = {}
    metrics["translation"] = calculate_affine_invariance(iter(dl), G, transform_translate)
    transform_rotate = StyleGANAugmentPipe(
        xflip=0.0,
        xint=0, xint_max=0.125,
        scale=0, scale_std=0.2,
        rotate90=0,
        rotate=1, rotate_max=0.25,
        aniso=0, aniso_std=0.2,
        xfrac=0, xfrac_std=0.125
    )
    metrics["rotation"] = calculate_affine_invariance(iter(dl), G, transform_rotate)
    
    transform_hflip = StyleGANAugmentPipe(
        xflip=1.0,
        xint=0, xint_max=0.125,
        scale=0, scale_std=0.2,
        rotate90=0,
        rotate=0, rotate_max=0.25,
        aniso=0, aniso_std=0.2,
        xfrac=0, xfrac_std=0.125
    )
    metrics["hflip"] = calculate_affine_invariance(iter(dl), G, transform_hflip)
    
    metrics = {f"metrics/affine_invariance/{key}": value for key, value in metrics.items()}
    logger.init(cfg, resume=True)
    logger.update_global_step(global_step)
    logger.log_dictionary(metrics)
    logger.finish()


if __name__ == "__main__":
    import os
    if os.environ.get("AMP_ENABLED") is None:
        print("AMP not set. setting to False")
        os.environ["AMP_ENABLED"] = "0"
    else:
        assert os.environ["AMP_ENABLED"] in ["0", "1"]
    main()

