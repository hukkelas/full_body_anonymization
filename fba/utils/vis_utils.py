import typing
import cv2
from kornia.morphology.morphology import erosion
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, List, Tuple,Optional
import torch
from fba.utils import download_file
from kornia.morphology import dilation
from torchvision.transforms.functional import resize, InterpolationMode
from fba.utils import from_E_to_vertex
from fba import utils

TOPLEFT = 0
BOTTOMOLEFT = 1


colors = list(matplotlib.colors.cnames.values())


def hex_to_rgb(h):
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


colors = [hex_to_rgb(x[1:]) for x in colors]
colors = [(255, 0, 0)] + colors


def draw_faces_with_keypoints(
        im: np.ndarray,
        im_bboxes: typing.Iterable,
        im_keypoints: typing.Iterable,
        radius: int = None, black_out_face=False, color_override: tuple = None
    ):
    im = im.copy()
    if im_keypoints is None:
        assert im_bboxes is not None, "Image bboxes cannot be None."
        im_keypoints = [None for i in range(len(im_bboxes))]
    if im_bboxes is None:
        im_bboxes = [None for i in range(len(im_keypoints))]
    if radius is None:
        radius = max(int(max(im.shape) * 0.0025), 1)
    for c_idx, (bbox, keypoint) in enumerate(zip(im_bboxes, im_keypoints)):
        color = color_override
        if color_override is None:
            color = colors[c_idx % len(colors)]

        if bbox is not None:
            x0, y0, x1, y1 = bbox
            if black_out_face:
                im[y0:y1, x0:x1, :] = 0
            else:
                im = cv2.rectangle(im, (x0, y0), (x1, y1), color)
        if keypoint is None:
            continue
        for x, y in keypoint:
            im = cv2.circle(im, (int(x), int(y)), radius, color)
    if not isinstance(im, np.ndarray):
        return im.get()
    return im


def np_make_image_grid(images, nrow, pad=2, row_major=True, pad_value=0):
    height, width = images[0].shape[:2]
    ncol = int(np.ceil(len(images) / nrow))
    for idx in range(len(images)):
        assert images[idx].shape == images[0].shape, (images[idx].shape, images[0].shape, idx)
    if isinstance(pad, int):
        pad = (pad, pad)
    
    if not row_major:
        t = nrow
        nrow = ncol
        ncol = t
    im_result = np.zeros(
        (nrow * (height + pad[0]), ncol * (width + pad[1]), images[0].shape[-1]), dtype=images[0].dtype
    ) + pad_value
    im_idx = 0
    for row in range(nrow):
        for col in range(ncol):
            if im_idx == len(images):
                break
            im = images[im_idx]
            if not row_major:
                im = images[row + col*nrow]
            im_idx += 1
            rstart = row * (pad[0] + height)
            rend = row * (pad[0] + height) + height
            cstart = col * (pad[1] + width)
            cend = col * (pad[1] + width) + width
            im_result[rstart:rend, cstart:cend, :] = im
    return im_result


def add_text(
        im: np.ndarray, x, y, text,
        font_scale: float = .7,
        line_thickness: float = 1,
        corner=None,
        fontColor = (255, 255, 255)):
    """
    Annotates text where (x,y) indicates top left corner.
    """
    im = np.ascontiguousarray(im)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if corner is None:
        bottomLeftCornerOfText = (x, y + int(25*font_scale))
    elif corner == TOPLEFT:
        bottomLeftCornerOfText = (0, 0 + int(25*font_scale))
    elif corner == BOTTOMOLEFT:
        bottomLeftCornerOfText = (0, im.shape[0]-1)
    else:
        raise ValueError(f"Not valid position {corner}")
    backgroundColor = (0, 0, 0)

    cv2.putText(
        im, text, bottomLeftCornerOfText, font, font_scale, backgroundColor,
        int(line_thickness * 2))
    cv2.putText(im, text, bottomLeftCornerOfText, font, font_scale, fontColor,
        line_thickness)
    return im


def add_label_y(im, positions, labels):
    # positions [(x, y)]
    im = im.copy()
    assert len(positions) == len(labels)
    for pos, label in zip(positions, labels):
        add_text(im, 0, pos, label)
    return im


def plot_bbox(bbox):
    x0, y0, x1, y1 = bbox
    plt.plot([x0, x0, x1, x1, x0], [y0, y1, y1, y0, y0])


def pad_im_as(im, target_im):
    assert len(im.shape) == 3
    assert len(target_im.shape) == 3
    assert im.shape[0] <= target_im.shape[0]
    assert im.shape[1] <= target_im.shape[1], f"{im.shape}, {target_im.shape}"
    pad_h = abs(im.shape[0] - target_im.shape[0]) // 2
    pad_w = abs(im.shape[1] - target_im.shape[1]) // 2
    im = np.pad(im, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))
    assert im.shape == target_im.shape
    return im

# IMplemented in Torchvision 10.0
@torch.no_grad()
def draw_segmentation_masks(
    image: torch.Tensor,
    masks: torch.Tensor,
    alpha: float = 0.8,
    colors: Optional[List[Union[str, Tuple[int, int, int]]]] = None,
) -> torch.Tensor:

    """
    Draws segmentation masks on given RGB image.
    The values of the input image should be uint8 between 0 and 255.

    Args:
        image (Tensor): Tensor of shape (3, H, W) and dtype uint8.
        masks (Tensor): Tensor of shape (num_masks, H, W) or (H, W) and dtype bool.
        alpha (float): Float number between 0 and 1 denoting the transparency of the masks.
            0 means full transparency, 1 means no transparency.
        colors (list or None): List containing the colors of the masks. The colors can
            be represented as PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
            When ``masks`` has a single entry of shape (H, W), you can pass a single color instead of a list
            with one element. By default, random colors are generated for each mask.

    Returns:
        img (Tensor[C, H, W]): Image Tensor, with segmentation masks drawn on top.
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"The image must be a tensor, got {type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"The image dtype must be uint8, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size()[0] != 3:
        raise ValueError("Pass an RGB image. Other Image formats are not supported")
    if masks.ndim == 2:
        masks = masks[None, :, :]
    if masks.ndim != 3:
        raise ValueError("masks must be of shape (H, W) or (batch_size, H, W)")
    if masks.dtype != torch.bool:
        raise ValueError(f"The masks must be of dtype bool. Got {masks.dtype}")
    if masks.shape[-2:] != image.shape[-2:]:
        raise ValueError("The image and the masks must have the same height and width")
    num_masks = masks.size()[0]
    if num_masks == 0:
        return image
    if not isinstance(colors[0], (Tuple, List)):
        colors = [colors for i in range(num_masks)]
    if colors is not None and num_masks > len(colors):
        raise ValueError(f"There are more masks ({num_masks}) than colors ({len(colors)})")

    if colors is None:
        colors = _generate_color_palette(num_masks)
    if not isinstance(colors, list):
        colors = [colors]
    if not isinstance(colors[0], (tuple, str)):
        raise ValueError("colors must be a tuple or a string, or a list thereof")
    if isinstance(colors[0], tuple) and len(colors[0]) != 3:
        raise ValueError("It seems that you passed a tuple of colors instead of a list of colors")

    out_dtype = torch.uint8

    colors_ = []
    for color in colors:
        if isinstance(color, str):
            color = ImageColor.getrgb(color)
        color = torch.tensor(color, dtype=out_dtype, device=masks.device)
        colors_.append(color)
    img_to_draw = image.detach().clone()
    # TODO: There might be a way to vectorize this
    for mask, color in zip(masks, colors_):
        img_to_draw[:, mask] = color[:, None]

    out = image * (1 - alpha) + img_to_draw * alpha
    return out.to(out_dtype)


def draw_boxes(
        im: np.ndarray,
        im_bboxes: typing.Iterable,
        color_override=None
        ):
    im = im.copy()
    for c_idx, bbox in enumerate(im_bboxes):
        color = color_override
        if color_override is not None:
            color = colors[c_idx % len(colors)]
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            im = cv2.rectangle(im, (x0, y0), (x1, y1), color)
    if not isinstance(im, np.ndarray):
        return im.get()
    return im


def _generate_color_palette(num_masks):
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    return [tuple((i * palette) % 255) for i in range(num_masks)]


def get_colormap(colormap=cv2.COLORMAP_JET):
    i = np.arange(0, 256, dtype=np.uint8).reshape(1, -1)
    return cv2.applyColorMap(i, colormap)[0]


color_embed_map = None

@torch.no_grad()
def get_cse_vertx2color_map():
    global color_embed_map
    if color_embed_map is not None:
        return color_embed_map
    color_embed_map, _ = np.load(download_file("https://dl.fbaipublicfiles.com/densepose/data/cse/mds_d=256.npy"), allow_pickle=True)
    color_embed_map = torch.from_numpy(color_embed_map).float()[:, 0]
    color_embed_map -= color_embed_map.min()
    color_embed_map /= color_embed_map.max()
    return color_embed_map


colormap_JET = None # torch.from_numpy(get_colormap(cv2.COLORMAP_JET))
def _init_colormap():
    global colormap_JET
    colormap_JET = torch.from_numpy(get_colormap(cv2.COLORMAP_JET))


def visualize_E(E, mask):
    E = E[:, 0]
    min_ = (E).min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    max_ = (E).max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    mask = mask.bool()
    E = (E - min_) / (max_ - min_)
    E = (E*255).long()
    E = (colormap_JET[E.cpu()]*(1-mask.byte().cpu()[:, 0, :, :, None])).numpy()
    return E


@torch.no_grad()
def visualize_vertices(vertices, mask):
    _init_colormap()
    assert len(vertices) == len(mask), (vertices.shape, mask.shape)
    assert vertices.shape[-2:] == mask.shape[-2:]
    colormap = (get_cse_vertx2color_map()*255).long()
    colormap = colormap_JET[colormap.cpu()][vertices.long().cpu()]
    colormap = colormap*(1-mask.byte().cpu().view(-1, *mask.shape[-2:], 1))
    return colormap.numpy()


@torch.no_grad()
def visualize_vertices_on_image(vertices, mask, img, t=0.8, **kwargs):
    mask = mask.float()
    colormap = visualize_vertices(vertices, mask).squeeze()
    assert isinstance(img, np.ndarray), "Expected numpy image"
    mask = (1 - mask).squeeze()[:, :, None].repeat(1, 1, 3).cpu().bool().numpy()
    img = img.copy()
    img[mask] = img[mask] * (1-t) + t * colormap[mask]
    return img

@torch.no_grad()
def visualize_vertices_torch(vertices, segmentation):
    _init_colormap()
    vertices = vertices.view(-1, *vertices.shape[-2:])
    segmentation = segmentation.view(-1, 1, *segmentation.shape[-2:])
    colormap = (get_cse_vertx2color_map()*255).long()
    # This operation might be good to do on cpu...
    colormap = colormap_JET[colormap][vertices.long()]
    colormap = colormap.to(segmentation.device)
    colormap = colormap.permute(0, 3, 1, 2)
    colormap = colormap*segmentation.byte()
    return colormap


def crop_box(x: torch.Tensor, bbox_XYXY) -> torch.Tensor:
    """
        Crops x by bbox_XYXY. 
    """
    x0, y0, x1, y1 = bbox_XYXY
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, x.shape[-1])
    y1 = min(y1, x.shape[-2])
    return x[..., y0:y1, x0:x1]


def remove_pad(x: torch.Tensor, bbox_XYXY, imshape):
    """
    Remove padding that is shown as negative 
    """
    H, W = imshape
    x0, y0, x1, y1 = bbox_XYXY
    padding = [
        max(0, -x0),
        max(0, -y0),
        max(x1 - W, 0),
        max(y1 - H, 0)
    ]
    x0, y0 = padding[:2]
    x1 = x.shape[2] - padding[2]
    y1 = x.shape[1] - padding[3]
    return x[:, y0:y1, x0:x1]


def draw_cse(
        embedding: torch.Tensor, E_mask: torch.Tensor, im: torch.Tensor,
        embed_map: torch.Tensor, vertices=None, t=0.7):
    """
        E_mask: 1 for areas with embedding
    """
    assert im.dtype == torch.uint8
    im = im.view(-1, *im.shape[-3:])
    E_mask = E_mask.view(-1, 1, *E_mask.shape[-2:])

    if vertices is None:
        embedding = embedding.view(-1, *embedding.shape[-3:])
        vertices = from_E_to_vertex(embedding, E_mask.logical_not().float(), embed_map)

    E_color = visualize_vertices_torch(vertices.squeeze(), E_mask)
    m = E_mask.bool().repeat(1, 3, 1, 1)
    im[m] = (im[m] * (1-t) + t * E_color[m]).byte()
    return im


def draw_cse_all(
        embedding: List[torch.Tensor], E_seg: List[torch.Tensor],
        im: torch.Tensor, boxes_XYXY: list, embed_map: torch.Tensor, t=0.7):
    """
        E_seg: 1 for areas with embedding
    """
    assert len(im.shape) == 3, im.shape
    assert im.dtype == torch.uint8

    N = len(E_seg)
    im = im.clone()
    for i in range(N):
        assert len(E_seg[i].shape) == 2
        assert len(embedding[i].shape) == 3
        assert embed_map.shape[1] == embedding[i].shape[0]
        assert len(boxes_XYXY[i]) == 4
        E = embedding[i]
        x0, y0, x1, y1 = boxes_XYXY[i]
        E = resize(E, (y1-y0, x1-x0), antialias=True)
        s = E_seg[i].float()
        s = (resize(s.squeeze()[None], (y1-y0, x1-x0), antialias=True) > 0).float()
        vertices = from_E_to_vertex(E[None], 1 - s[None], embed_map)
        E_color = visualize_vertices_torch(vertices.squeeze(), s.squeeze())
        s = s.bool().repeat(3, 1, 1)
        box = boxes_XYXY[i]
        s = remove_pad(s, box, im.shape[1:])
        E_color = remove_pad(E_color[0], box, im.shape[1:])
        E_color = E_color.to(im.device)
        crop_box(im, box)[s] = (crop_box(im, box)[s] * (1-t) + t * E_color[s]).byte()
    return im


def draw_mask(im: torch.Tensor, mask: torch.Tensor, t=0.2, color=(255, 255, 255), visualize_instances=True):
    """
        Visualize mask where mask = 0.
        Supports multiple instances.
        mask shape: [N, C, H, W], where C is different instances in same image.
    """
    orig_imshape = im.shape
    if mask.numel() == 0: return im
    assert len(mask.shape) in (3, 4), mask.shape
    mask = mask.view(-1, *mask.shape[-3:])
    im = im.view(-1, *im.shape[-3:])
    assert im.dtype == torch.uint8, im.dtype
    assert 0 <= t <= 1
    if not visualize_instances:
        mask = mask.any(dim=1, keepdim=True)
    mask = mask.float()
    kernel = torch.ones((3, 3), dtype=mask.dtype, device=mask.device)
    outer_border = dilation(mask, kernel).logical_xor(mask)
    outer_border = outer_border.any(dim=1, keepdim=True).repeat(1, 3, 1, 1) > 0
    inner_border = erosion(mask, kernel).logical_xor(mask)
    inner_border = inner_border.any(dim=1, keepdim=True).repeat(1, 3, 1, 1) > 0
    mask = (mask == 0).any(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    color = torch.tensor(color).to(im.device).byte().view(1, 3, 1, 1)#.repeat(1, *im.shape[1:])
    color = color.repeat(im.shape[0], 1, *im.shape[-2:])
    im[mask] = (im[mask] * (1-t) + t * color[mask]).byte()
    im[outer_border] = 255
    im[inner_border] = 0
    return im.view(*orig_imshape)


def draw_cropped_masks(im: torch.Tensor, mask: torch.Tensor, boxes: torch.Tensor, **kwargs):
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = boxes[i]
        orig_shape = (y1-y0, x1-x0)
        m = resize(mask[i], orig_shape, InterpolationMode.NEAREST).squeeze()[None]
        m = remove_pad(m, boxes[i], im.shape[-2:])
        crop_box(im, boxes[i]).set_(draw_mask(crop_box(im, boxes[i]), m))
    return im

def visualize_batch(
        img: torch.Tensor, mask: torch.Tensor,
        vertices: torch.Tensor=None, E_mask: torch.Tensor=None,
        embed_map: torch.Tensor=None,
        semantic_mask: torch.Tensor=None, **kwargs) -> torch.ByteTensor:
    img = utils.denormalize_img(img).mul(255).byte()
    img = draw_mask(img, mask)
    if vertices is not None:
        assert E_mask is not None
        assert embed_map is not None
        img = draw_cse(None, E_mask, img, embed_map, vertices)
    elif semantic_mask is not None:
        img = draw_segmentation_masks(img, semantic_mask)
    return img
    
    
