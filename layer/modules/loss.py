import torch
import torch.nn as nn
from functools import reduce
from itertools import product
from math import sqrt
import numpy as np

from ..function import iou_xywh

f_map = ((38, 512, 4, (2)),
        (19, 1024, 6, (2, 3)),
        (10, 512, 6, (2, 3)),
        (5, 256, 6, (2, 3)),
        (3, 256, 4, (2)),
        (1, 128, 4, (2)))
num_classes = 21

class SSDLoss(nn.Module):
    """
    loss of single shot detector
    """

    def __init__(self, ):
        super(SSDLoss, self).__init__()

    def forward(self, prediction, target):
        """
        forward(prediction, target) -> loss

        prediction : ( loc score, conf score, default box )
        loc score shape : (N, default box num, 4)
        conf score shape : ( N, default box num, num_classes )
        default box : ( default box num, (cx,cy,w,h)(all 0~1) )

        target : ( N, [truthbox, class] )

        return : loss of ssd
        """
        # seperate scores
        loc = prediction[0]
        conf = prediction[1]
        defaultbox = prediction[2]

        truthbox = target[:,:-1]
        truthclasses = target[:,-1]

        #matching defaultbox


        #negative mining

        locloss =
        confloss =

        return torch.mean(locloss), torch.mean(confloss)
