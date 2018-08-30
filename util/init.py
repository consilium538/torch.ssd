import torch;import torch.nn as nn
import torch.nn.init as init

def xavier(x):
    init.xavier_uniform_(x)

def init_xavier(net):
    if isinstance(i, nn.Conv2D):
        xavier(i.weight.data)
