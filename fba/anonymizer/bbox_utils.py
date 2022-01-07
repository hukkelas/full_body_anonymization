from copy import deepcopy
import numpy as np


def get_bbox_segmentation(S, box_xywh):
    S = S.astype(float)
    def find_first_nonzero(array, reverse):
        for i in range(array.shape[0]):
            if reverse:
                i = - i - 1
            if array[i] != 0:
                if reverse:
                    return min(array.shape[0] + i + 2, array.shape[0]) # can be +1 if at end of array
                return i
        if reverse:
            return array.shape[0]
        return 0
    x0 = find_first_nonzero(S.sum(axis=0), False)
    x1 = find_first_nonzero(S.sum(axis=0), True)

    y0 = find_first_nonzero(S.sum(axis=1), False)
    y1 = find_first_nonzero(S.sum(axis=1), True)
    x0 += box_xywh[0]
    x1 += box_xywh[0]
    y0 += box_xywh[1]
    y1 += box_xywh[1]
    return [int(_) for _ in [x0, y0, x1, y1]]


def expand_bbox_to_ratio(bbox, imshape, target_aspect_ratio):
    x0, y0, x1, y1 = [int(_) for _ in bbox]
    h, w = y1 - y0, x1 - x0
    cur_ratio = h / w

    if cur_ratio == target_aspect_ratio:
        return [x0, y0, x1, y1]
    if cur_ratio < target_aspect_ratio:
        target_height = int(w*target_aspect_ratio)
        y0, y1 = expand_axis(y0, y1, target_height, imshape[0])
    else:
        target_width = int(h/target_aspect_ratio)
        x0, x1 = expand_axis(x0, x1, target_width, imshape[1])
    return x0, y0, x1, y1


def expand_axis(start, end, target_width, limit):
#    print(start, end, target_width, limit)
    # Can return a bbox outside of limit
    cur_width = end - start
    start = start - (target_width-cur_width)//2
    end = end + (target_width-cur_width)//2
    if end - start != target_width:
        end += 1
    assert end - start == target_width
    if start < 0 and end > limit:
        return start, end
    if start < 0 and end < limit:
        to_shift = min(0 - start, limit - end)
        start += to_shift
        end += to_shift
    if end > limit and start > 0:
        to_shift = min(end - limit, start)
        end -= to_shift
        start -= to_shift
    assert end - start == target_width
    return start, end


def expand_box(bbox, imshape, S, S_box, percentage_background: float):
    assert isinstance(bbox[0], int)
    assert 0 < percentage_background < 1
    # Percentage in S
    percentage = S.astype(int).mean() # S is 112x112 but correspond to the area size of S_box_XYWH
    num_pixels = int((S_box[2] - S_box[0]) * (S_box[3] - S_box[1]) * percentage)
    target_pixels = int(num_pixels/(1-percentage_background))
    cur_pixels = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    if cur_pixels >= target_pixels:
        return bbox

    x0, y0, x1, y1 = bbox
    H = y1 - y0
    W = x1 - x0
    p = np.sqrt(target_pixels/(H*W))
    target_width = int(p * W)
    target_height = int(p * H)
    x0, x1 = expand_axis(x0, x1, target_width, imshape[1])
    y0, y1 = expand_axis(y0, y1, target_height, imshape[0])
    return [x0, y0, x1, y1]


def expand_axises_by_percentage(bbox_XYXY, imshape, percentage):
    x0, y0, x1, y1 = bbox_XYXY
    x0, x1 = expand_axis(x0, x1, min(int((x1-x0)*(1+percentage)), imshape[1]), imshape[1])
    y0, y1 = expand_axis(y0, y1, min(int((y1-y0)*(1+percentage)), imshape[0]), imshape[0])
    return [x0, y0, x1, y1]


def get_surrounding_bbox(
        mask,
        bbox,
        imshape,
        percentage_background: float,
        axis_minimum_expansion: float,
        target_aspect_ratio: float
    ):

    assert mask.shape[0] == int(bbox[3] - bbox[1]), (mask.shape, imshape)
    assert mask.shape[1] == int(bbox[2] - bbox[0]), (mask.shape, bbox)

    bbox_XYXY = get_bbox_segmentation(mask, bbox)
    orig_bbox = deepcopy(bbox_XYXY)
    # Expand each axis of the bounding box by a minimum percentage
    bbox_XYXY = expand_axises_by_percentage(bbox_XYXY, imshape, axis_minimum_expansion) 
    # Find the minimum bbox with the aspect ratio. Can be outside of imshape
    bbox_XYXY = expand_bbox_to_ratio(bbox_XYXY, imshape, target_aspect_ratio)
    # Expands square box such that X% of the bbox is background
    bbox_XYXY = expand_box(bbox_XYXY, imshape, mask,  bbox, percentage_background)
#    print((bbox_XYXY[2]-bbox_XYXY[0])/(bbox_XYXY[3] - bbox_XYXY[1]))
    assert isinstance(bbox_XYXY[0], int)
    return bbox_XYXY, orig_bbox


def annotate_expanded_bbox(cse_prediction, imshape, percentage_background, axis_minimum_expansion, target_aspect_ratio):
    for pred in cse_prediction:
        exp_bbox, orig_bbox = get_surrounding_bbox(pred, imshape, percentage_background, axis_minimum_expansion, target_aspect_ratio)
        pred["expanded_bbox"] = exp_bbox
        pred["orig_bbox"] = orig_bbox


def include_box(bbox, minimum_area, aspect_ratio_range, min_bbox_ratio_inside, imshape):
    def area_inside_ratio(bbox, imshape):
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        area_inside = (min(bbox[2], imshape[1]) - max(0,bbox[0])) * (min(imshape[0],bbox[3]) - max(0,bbox[1]))
        return area_inside / area
    ratio = (bbox[3] - bbox[1]) / (bbox[2] - bbox[0])
    area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
    if area_inside_ratio(bbox, imshape) < min_bbox_ratio_inside:
        return False
    if ratio <= aspect_ratio_range[0] or ratio >= aspect_ratio_range[1] or area < minimum_area:
        return False
    return True