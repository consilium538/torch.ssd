#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch;import torch.nn as nn;import torch.nn.functional as F
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from layer.modules.ssd import SSD
from layer.modules.loss import SSDLoss
from layer.function.iou import match
from data.VOC0712 import VOCDataset

# load weight
net = SSD()
if os.path.isfile(arg['weightpath']):
    net.load_state_dict(torch.load(arg['wegihtpath']))
    print('Pervious trainded weight loaded')

# dataloader
dataset = VOCDataset('./') # imgaug transform 지원 하게 수정
dataloader = data.DataLoader(dataset, arg['batchsize'])
batch_iterator = iter(dataloader)

# for i in range(epoch)
for batch in range(arg['startiter'], arg['enditer']):
    img, label = next(batch_iterator)
    result = ssd(img)
    optim.zero_grad()
    loss = SSDLoss(result, label)
    loss.backward()
    optim.step()
#   for j in dataloader
#       a = ssd(j)
#       loss = criterion(a,j)
#       optim.zero_grad()
#       loss.backword
#       optim.step()
#   save weight
