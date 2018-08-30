import torch;import torch.nn as nn
import torch.nn.init as init

def xavier(x):
    init.xavier_uniform_(x)

def init_xavier(param):
    if isinstance(param, nn.Conv2d):
        xavier(param.weight.data)
