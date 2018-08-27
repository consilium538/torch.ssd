import torch
import numpy as np

def lse(x):
    """
    logsumexp function
    """
    x_max = x.max()
    return torch.log(torch.sum(torch.exp(x-x_max),1,keepdim=True)) + x_max
