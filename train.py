import torch;import torch.nn as nn;import torch.nn.functional as F
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from layer.modules.ssd import SSD
from layer.modules.loss import SSDLoss
from layer.function.iou import match


