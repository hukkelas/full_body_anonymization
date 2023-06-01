import torch
import torchvision
from typing import List
from fba import utils
from torchvision.transforms.functional import InterpolationMode, resize
from densepose.data.utils import get_class_to_mesh_name_mapping
from densepose import add_densepose_config
from densepose.structures import DensePoseEmbeddingPredictorOutput
from densepose.vis.extractor import DensePoseOutputsExtractor
from densepose.modeling import build_densepose_embedder
from detectron2.config import get_cfg
from detectron2.data.transforms import ResizeShortestEdge
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model


model_urls = {
    "https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/DensePose/configs/cse/densepose_rcnn_R_101_FPN_DL_soft_s1x.yaml": "https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_101_FPN_DL_soft_s1x/250713061/model_final_1d3314.pkl",
    "https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/DensePose/configs/cse/densepose_rcnn_R_50_FPN_s1x.yaml": "https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_s1x/251155172/model_final_c4ea5f.pkl",
}


def cse_det_to_global(boxes_XYXY, S: torch.Tensor, imshape):
    assert len(S.shape) == 3
    H, W = imshape
    N = len(boxes_XYXY)
    segmentation = torch.zeros((N, H, W), dtype=torch.bool, device=S.device)
    boxes_XYXY = boxes_XYXY.long()
    for i in range(N):
        x0, y0, x1, y1 = boxes_XYXY[i]
        assert x0 >= 0 and y0 >= 0
        assert x1 <= imshape[1]
        assert y1 <= imshape[0]
        h = y1 - y0
        w = x1 - x0
        segmentation[i:i+1, y0:y1, x0:x1] = resize(S[i:i+1], (h, w), interpolation=InterpolationMode.NEAREST) > 0
    return segmentation


class CSEDetector:

    def __init__(
            self,
            cfg_url: str = "https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/DensePose/configs/cse/densepose_rcnn_R_101_FPN_DL_soft_s1x.yaml",
            cfg_2_download: List[str] = [
                "https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/DensePose/configs/cse/densepose_rcnn_R_101_FPN_DL_soft_s1x.yaml",
                "https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/DensePose/configs/cse/Base-DensePose-RCNN-FPN.yaml",
                "https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/DensePose/configs/cse/Base-DensePose-RCNN-FPN-Human.yaml"],
            score_thres: float = 0.9,
            nms_thresh: float = None,
            ) -> None:
        cfg = get_cfg()
        self.device = utils.get_device()
        add_densepose_config(cfg)
        cfg_path = utils.download_file(cfg_url)
        for p in cfg_2_download:
            utils.download_file(p)
        cfg.merge_from_file(cfg_path)
        assert cfg_url in model_urls, cfg_url
        model_path = utils.download_file(model_urls[cfg_url])
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thres
        if nms_thresh is not None:
            cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_thresh
        cfg.MODEL.WEIGHTS = str(model_path)
        cfg.MODEL.DEVICE = str(self.device)
        cfg.freeze()
        self.model = build_model(cfg)
        self.model.eval()
        DetectionCheckpointer(self.model).load(str(model_path))
        self.input_format = cfg.INPUT.FORMAT
        self.densepose_extractor = DensePoseOutputsExtractor()
        self.class_to_mesh_name = get_class_to_mesh_name_mapping(cfg)

        self.embedder = build_densepose_embedder(cfg)
        self.mesh_vertex_embeddings = {
            mesh_name: self.embedder(mesh_name).to(self.device)
            for mesh_name in self.class_to_mesh_name.values()
            if self.embedder.has_embeddings(mesh_name)
        }
        self.cfg = cfg

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def resize_im(self, im):
        H, W = im.shape[1:]
        newH, newW = ResizeShortestEdge.get_output_shape(
            H, W, self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MAX_SIZE_TEST)
        return resize(
            im, (newH, newW), InterpolationMode.BILINEAR, antialias=True)

    @torch.no_grad()
    def forward(self, im):
        assert im.dtype == torch.uint8
        if self.input_format == "BGR":
            im = im.flip(0)
        H, W = im.shape[1:]
        im = self.resize_im(im)
        output = self.model([{"image": im, "height": H, "width": W}])[0]["instances"]
        scores = output.get("scores")
        if len(scores) == 0:
            return None
        pred_densepose, boxes_xywh, classes = self.densepose_extractor(output)
        assert isinstance(pred_densepose, DensePoseEmbeddingPredictorOutput), pred_densepose
        S = pred_densepose.coarse_segm.argmax(dim=1) # Segmentation channel Nx2xHxW (2 because only 2 classes)
        E = pred_densepose.embedding
        mesh_name = self.class_to_mesh_name[classes[0]]
        assert mesh_name == "smpl_27554"
        embed_map = self.mesh_vertex_embeddings[mesh_name]
        x0, y0, w, h = [boxes_xywh[:, i] for i in range(4)]
        boxes_XYXY = torch.stack((x0, y0, x0+w, y0+h), dim=-1)
        boxes_XYXY = boxes_XYXY.round_().long()

        non_empty_boxes = (boxes_XYXY[:, :2] == boxes_XYXY[:, 2:]).any(dim=1).logical_not()
        S = S[non_empty_boxes]
        E = E[non_empty_boxes]
        boxes_XYXY = boxes_XYXY[non_empty_boxes]

        im_segmentation = cse_det_to_global(boxes_XYXY, S, [H, W])
        return dict(
            instance_segmentation=S, instance_embedding=E,
            embed_map=embed_map, bbox_XYXY=boxes_XYXY,
            im_segmentation=im_segmentation)



if __name__ == "__main__":
    import click
    from PIL import Image
    from fba.utils import from_E_to_vertex
    from fba.utils.vis_utils import visualize_vertices_on_image
    @click.command()
    @click.argument("input_path")
    def main(input_path):
        im = torchvision.io.read_image(input_path, mode=torchvision.io.ImageReadMode.RGB)
        detector = CSEDetector()
        detections = detector(im)
        embed_map = detections["embed_map"]
        im = utils.image2np(im)
        for i, bbox in enumerate(detections["bbox_XYXY"]):
            s = detections["instance_segmentation"][i]
            E = detections["instance_embedding"][i]
            x0, y0, x1, y1 = bbox

            E = resize(E, (y1-y0, x1-x0), antialias=True)
            s = resize(s.float()[None], (y1-y0, x1-x0), antialias=True) > 0
            vertices = from_E_to_vertex(E[None], s[None].float().logical_not_(), embed_map)
            im[y0:y1, x0:x1] = visualize_vertices_on_image(
                vertices, s.squeeze().logical_not_(), im[y0:y1, x0:x1])

        im = Image.fromarray(im)
        im.save("test.png")
        im.show()
    main()
            


