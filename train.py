#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch;import torch.nn as nn;import torch.nn.functional as F
from torch.optim import Adam
from torch.utils import data
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from layer.modules.ssd import SSD
from layer.modules.loss import SSDLoss
from util.init import init_xavier
from data.voc0712 import VOCDataset, detect_collate
import imgaug as ia
from imgaug import augmenters as iaa

from config import arg

import os
import time

print('\n'+'-'*15+'*'+'-'*15+'\n')
print('start train')
# load weight or init
net = SSD().cuda()
#if os.path.isfile(arg['weightpath']):
    #net.load_state_dict(torch.load(arg['weightpath']))
    #print('Pervious trainded weight loaded')
#else:
net.extra.apply(init_xavier)
net.mbox_conf.apply(init_xavier)
net.mbox_loc.apply(init_xavier)
print('Initalize model')

# dataloader, loss, optim
dataset = VOCDataset('./',iaa.Scale(300)) # imgaug transform 지원 하게 수정
dataloader = data.DataLoader(dataset, arg['batchsize'], num_workers=0,
        shuffle=True, collate_fn=detect_collate)
print('dataloader ready')
criterion = SSDLoss()
optim = Adam([
    *net.extra.parameters(),
    *net.mbox_conf.parameters(),
    *net.mbox_loc.parameters()
    ], lr=arg['lr'])
print('loss, optim ready')
print('\n'+'-'*15+'*'+'-'*15+'\n')

# for i in range(epoch)
for epoch in range(arg['term_epoch']):
    batch_iterator = iter(dataloader)
    at = time.time()
    batch_iter = 0
    for batch in batch_iterator:
        batch_iter += 1
        img, anno = batch
        optim.zero_grad()
        hypothesis = net(img)
        loss_l, loss_c = criterion(hypothesis, anno)
        loss = loss_l + loss_c
        loss.backward()
        optim.step()
        if batch_iter % 100 == 0:
            print(loss)
    print(time.time() - at)
    torch.save({
        'extra':net.extra.state_dict(),
        'mbox_conf':net.mbox_conf.state_dict(),
        'mbox_loc':net.mbox_loc.state_dict()
        }, arg['weightpath'])
    print('one epoch end')
#   for j in dataloader
#       a = ssd(j)
#       loss = criterion(a,j)
#       optim.zero_grad()
#       loss.backword
#       optim.step()
#   save weight
