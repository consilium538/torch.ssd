import torch
import numpy as np

def iou_xywh(bbox1, bbox2):
    """
    calculate intersection over union(IoU)
    input : bbox1, bbox2
    bbox : (cx, cy, w, h)

    output : IoU
    """
    #area1 = area_xyxy(xywh_to_xyxy(bbox1))
    #area2 = area_xyxy(xywh_to_xyxy(bbox2))

    bbox_xyxy = (xywh_to_xyxy(bbox1), xywh_to_xyxy(bbox2))
    area1 = area_xyxy(bbox_xyxy[0])
    area2 = area_xyxy(bbox_xyxy[1])
    iou = intersection_xyxy(*bbox_xyxy)
    if area1 == 0 or area2 == 0:
        return
    return iou / (area1 + area2 - iou)

def xywh_to_xyxy(bbox):
    return (bbox[0] - bbox[2] / 2, bbox[1] - bbox[3] / 2,
            bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)

def intersection_xyxy(bbox1, bbox2):
    """
    intersection calculation
    """
    up = max(bbox1[0], bbox2[0])
    down = min(bbox1[2], bbox2[2])
    left = max(bbox1[1], bbox2[1])
    right = min(bbox1[3], bbox2[3])
    # no intersection case
    if up > down or left > right:
        return 0

    return (down - up) * (right - left)

def area_xyxy(bbox):
    if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
        return 0
    return ( bbox[2] - bbox[0] )*( bbox[3] - bbox[1] )
