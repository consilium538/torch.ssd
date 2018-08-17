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

def c2p(bbox):
    """
    convert bboxs format from ( x centor, y centor, width, hight )
    to ( x min, y min, x max, y max )
    """
    return (bbox[:,0] - bbox[:,2] / 2, bbox[:,1] - bbox[:,3] / 2,
            bbox[:,0] + bbox[:,2] / 2, bbox[:,1] + bbox[:,3] / 2)

def p2c(bbox):
    """
    convert bboxs format from ( x min, y min, x max, y max )
    to ( x centor, y centor, width, hight )
    """
    return (( bbox[:,0] + bbox[:,2] ) / 2, ( bbox[:,1] + bbox[:,3] ) / 2,
            ( bbox[:,2] - bbox[:,0] ) / 2, ( bbox[:,3] - bbox[:,1] ) / 2)

def intersection_xyxy(bbox1, bbox2):
    """
    intersection calculation
    """
    A = bbox1.size(0)
    B = bbox2.size(0)
    xymax = torch.max(
            bbox1[:,:2].unsqueeze(1).expand(A,B,2),
            bbox2[:,:2].unsqueeze(0).expand(A,B,2)
            )
    xymin = torch.min(
            bbox1[:,2:].unsqueeze(1).expand(A,B,2),
            bbox2[:,2:].unsqueeze(0).expand(A,B,2)
            )
    interlen = torch.clamp(xymax-xymin, min=0)
    return interlen[:,:,0] * interlen[:,:,1]

def iou(bbox1, bbox2):
    bbox1_p = c2p(bbox1)
    bbox2_p = c2p(bbox2)

    bbox1_size = ( bbox1[:,2] - bbox1[:,0] ) * ( bbox1[:,3] - bbox1[:,1] )
    bbox2_size = ( bbox2[:,2] - bbox2[:,0] ) * ( bbox2[:,3] - bbox2[:,1] )

    inter = intersection_xyxy(bbox1_p,bbox2_p)
    union = bbox1_size + bbox2_size - inter
    return inter / union

def match()
