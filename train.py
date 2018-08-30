#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch;import torch.nn as nn;import torch.nn.functional as F
from torch.optim import Adam
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from layer.modules.ssd import SSD
from layer.modules.loss import SSDLoss
from util.init import init_xavior
from data.voc0712 import VOCDataset, detect_collate

# load weight or init
net = SSD()
if os.path.isfile(arg['weightpath']):
    net.load_state_dict(torch.load(arg['wegihtpath']))
    print('Pervious trainded weight loaded')
else:
    net.extra.apply(init_xavior)
    net.mbox_conf.apply(init_xavior)
    net.mbox_loc.apply(init_xavior)

# dataloader, loss, optim
dataset = VOCDataset('./') # imgaug transform 지원 하게 수정
dataloader = data.DataLoader(dataset, arg['batchsize'], num_workers=4,
        shuffle=True, collate_fn=detect_collate, pin_memory=True)
batch_iterator = iter(dataloader)
criterion = SSDLoss()
optim = Adam([
    net.extra.parameters(),
    net.mbox_conf.parameters(),
    net.mbox_loc.parameters()
    ])

# for i in range(epoch)
for epoch in range(arg['term_epoch']):
    batch_iterator = iter(dataloader)
    for batch in batch_iterator:
        img, anno = batch
        optim.zero_grad()
        hypothesis = net(img)
        loss_l, loss_c = criterion(hypothesis, anno)
        loss = loss_l + loss_c
        loss.backword()
        optim.step()
    torch.save({
        'extra':net.extra.state_dict(),
        'mbox_conf':net.mbox_conf.state_dict(),
        'mbox_loc':net.mbox_loc.state_dict()
        }, f'./models/{arg["name"]}')
#   for j in dataloader
#       a = ssd(j)
#       loss = criterion(a,j)
#       optim.zero_grad()
#       loss.backword
#       optim.step()
#   save weight
