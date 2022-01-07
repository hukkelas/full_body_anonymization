import torch
from skimage.morphology import disk


@torch.no_grad()
def binary_dilation(im: torch.Tensor, kernel: torch.Tensor):
    assert len(im.shape) == 4
    assert len(kernel.shape) == 2
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    padding = kernel.shape[-1]//2
    assert kernel.shape[-1] % 2 != 0
    if torch.cuda.is_available():
        im, kernel = im.half(), kernel.half()
    else:
        im, kernel = im.float(), kernel.float()
    im = torch.nn.functional.conv2d(
        im, kernel, groups=im.shape[1], padding=padding)
    im = im.clamp(0, 1).bool()
    return im


def test_dilation():
    from skimage.morphology import binary_dilation as skimage_binary_dilation
    from kornia.morphology import dilation as kornia_dilation
    import numpy as np
    import time
    im = np.random.randint(0, 1, size=(512, 512)).astype(np.bool)
    sizes = [3, 9, 21, 91]
    for s in sizes:
        kernel_np = disk(s).astype(np.bool)
        kernel_torch = torch.from_numpy(kernel_np)
        im_torch = torch.from_numpy(im)[None, None]
        s = time.time()
        result_skimage = skimage_binary_dilation(im, kernel_np)
        print("Skimage", time.time() - s)
        s = time.time()
        result_kornia = kornia_dilation(im_torch.float(), kernel_torch.float()).bool().cpu().numpy().squeeze()
        print("Kornia", time.time() - s)
        s = time.time()
        result_ours = binary_dilation(im_torch, kernel_torch).cpu().numpy().squeeze()
        print("Ours", time.time() - s)
        np.testing.assert_almost_equal(result_skimage, result_kornia)
        np.testing.assert_almost_equal(result_skimage, result_ours)        


if __name__ == "__main__":
    test_dilation()
