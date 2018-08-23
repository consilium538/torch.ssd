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
        self.negpos_ratio = 3

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
        match_list = list()
        loc_list = list()

        #matching defaultbox : return index{1:positive,-1:negative,0:not both}
        for idx in range(num):
            truthbox = target[idx]
            box_loc, box_classes = match(defaultbox, truthbox)
            #box_classes.type() == 'torch.cuda.LongTensor'

            #negative mining will done in funtion match
            #should done in here...
            # dimension : [8732,num_classes]
            box_one_hot = torch.cuda.FloatTensor(box_classes.size(0),21)\
                    .zero_().scatter_(1,box_classes.unsqueeze(1),1.0)
            match_list.append(box_one_hot)
            loc_list.append(box_loc)

        matchbox = torch.stack(match_list) # [N, num_defaultbox,num_classes]
        loc_conf = torch.stack(loc_list) # [N, num_defaultbox]
        #negative_list = [negative_proposal[:,i] for i in range(num_classes)]

        num_pos = matchbox[:,:,:-1].long().sum(1)
        num_neg = torch.clamp(num_pos.sum(1),max=matchbox.size(1)-1)

        # loc loss : L1smooth of matched boxes

        # conf loss : CrossEntropy of POS group and NEG group

        #locloss = 0
        #confloss = 0
        #return torch.mean(locloss), torch.mean(confloss)
