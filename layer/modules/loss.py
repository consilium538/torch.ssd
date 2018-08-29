import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from itertools import product
from math import sqrt
import numpy as np

from ..function.iou import match
from ..function.lse import lse

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

    def __init__(self, negpos_ratio = 3):
        super(SSDLoss, self).__init__()
        self.negpos_ratio = negpos_ratio

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
        loc, conf, defaultbox = prediction

        matchbox = torch.cuda.LongTensor(*conf.shape[:-1]) # [N, num_defaultbox]
        loc_conf = torch.cuda.FloatTensor(*loc.shape) # [N, num_defaultbox, 4]

        for idx in range(loc.shape[0]):
            truthbox = target[idx]
            box_loc, box_classes = match(defaultbox, truthbox)

            matchbox[idx] = box_classes
            loc_conf[idx] = box_loc

        matchidx = torch.zeros(*conf.shape).scatter_(
                2,matchbox.unsqueeze(2),1.0
                ) # [N, num_defaultbox, num_classes]
        pos = (matchbox != 20)
        num_pos = pos.sum(dim=1, keepdim=True)
        num_neg = torch.clamp(num_pos * self.negpos_ratio, max=pos.size(1)-1)

        pos_idx = pos.unsqueeze(2).expand_as(loc)
        loc_p = loc[pos_idx].view(-1,4)
        loc_t = loc_conf[pos_idx].view(-1,4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        batch_conf = conf.view(-1,num_classes)
        loss_tmp = lse(batch_conf) - batch_conf.gather(1,matchbox.view(-1,1))
        loss_tmp = loss_tmp.view(-1,pos.shape[1])
        loss_tmp[pos] = 0
        _, conf_idx = torch.sort(loss_tmp, descending=True)
        _, rank_idx = torch.sort(conf_idx)
        neg = rank_idx < num_neg.expand_as(rank_idx)

        pos_idx = pos.unsqueeze(2).expand_as(conf)
        neg_idx = neg.unsqueeze(2).expand_as(conf)
        conf_p = conf[(pos_idx+neg_idx).gt(0)].view(-1,num_classes)
        target_weighted = matchbox[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, target_weighted, reduction='sum')

        N = num_pos.data.sum()
        loss_c /= N
        loss_l /= N
        return loss_l + loss_c
