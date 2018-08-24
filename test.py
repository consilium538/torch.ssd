import torch;import torch.nn as nn;import torch.nn.functional as F
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from layer.modules.ssd import SSD
from layer.modules.loss import SSDLoss
from layer.function.iou import match
dummy_img = torch.rand(2,3,300,300)
ssd_net = SSD()
lossf = SSDLoss()
conf = ssd_net(dummy_img)
dummy_truth = [torch.tensor((0.0,0.0,1.0,1.0,2.0)).reshape(1,-1),
     torch.tensor([(0.5,0.5,1.0,1.0,3.0),(0.0,0.0,1.0,1.0,2.0)]).reshape(2,-1)]
testing = lossf(conf, dummy_truth)

