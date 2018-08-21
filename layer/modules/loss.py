import torch
import torch.nn as nn
from functools import reduce
from itertools import product
from math import sqrt
import numpy as np

from ..function import match

f_map = ((38, 512, 4, (2,)),
        (19, 1024, 6, (2, 3)),
        (10, 512, 6, (2, 3)),
        (5, 256, 6, (2, 3)),
        (3, 256, 4, (2,)),
        (1, 128, 4, (2,)))
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

        target : ( N, num_truthbox, [truthbox, class] )

        return : loss of ssd
        """
        # seperate scores
        #loc = prediction[0]
        #conf = prediction[1]
        #defaultbox = prediction[2]
        loc, conf, defaultbox = prediction

        #matching defaultbox : return index{1:positive,-1:negative,0:not both}
        for idx in range(num):
            truthbox = target[idx][:,:-1]
            positive_box = match(defaultbox, truthbox[idx])

            #negative mining will done in funtion match
            #should done in here...
            # dimention : [8732,num_classes]

            negative_proposal = 1 - positive_box
            #negative_list = [negative_proposal[:,i] for i in range(num_classes)]

        #locloss = 0
        #confloss = 0
#
        #return torch.mean(locloss), torch.mean(confloss)
