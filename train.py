#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch;import torch.nn as nn;import torch.nn.functional as F
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from layer.modules.ssd import SSD
from layer.modules.loss import SSDLoss
from layer.function.iou import match

# load weight
net = SSD()
if os.path.isfile(arg['weightpath']):
    net.load_state_dict(torch.load(arg['wegihtpath']))
    print('Pervious trainded weight loaded')
else:
    net.vgg
# dataloader
# for i in range(epoch)
#   for j in dataloader
#       a = ssd(j)
#       loss = criterion(a,j)
#       optim.zero_grad()
#       loss.backword
#       optim.step()
#   save weight
