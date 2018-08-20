################################

vgg = torchvision.models.vgg16(pretrained=True)
for i in vgg.features:
    for j in i.parameters():
        j.requires_grad = False

################################

>>> vgg.features #input : 300 300
Sequential(
  (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU(inplace)
  (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (3): ReLU(inplace)
  (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) #150
  (6): ReLU(inplace)
  (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (8): ReLU(inplace)
  (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) #75
  (11): ReLU(inplace)
  (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (13): ReLU(inplace)
  (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (15): ReLU(inplace)
  (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) #37
  (18): ReLU(inplace)
  (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (20): ReLU(inplace)
  (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (22): ReLU(inplace)
  (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) #18
  (25): ReLU(inplace)
  (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (27): ReLU(inplace)
  (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (29): ReLU(inplace)
  (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) #9
)

################################

a = torchvision.models.vgg16().features[:-1]
for i in a:
    if isinstance(i,nn.MaxPool2d):
        i.ceil_mode = True

################################

vgg = torchvision.models.vgg16(pretrained=True).features[:-1]
for i in vgg
    if isinstance(i, nn.MaxPool2d):
        i.ceil_mode = True

extra = nn.ModuleList([ #19
    nn.Conv2d(512, 1024, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
    nn.Conv2d(1024, 1024, kernel_size=(1,1), stride=(1,1), padding=(0,0)), #19
    nn.Conv2d(1024, 256, kernel_size=(1,1), stride=(1,1), padding=(0,0)),
    nn.Conv2d(256, 512, kernel_size=(3,3), stride=(2,2), padding=(1,1)), #10
    nn.Conv2d(512, 128, kernel_size=(1,1), stride=(1,1), padding=(0,0)),
    nn.Conv2d(128, 256, kernel_size=(3,3), stride=(2,2), padding=(1,1)), #5
    nn.Conv2d(256, 128, kernel_size=(1,1), stride=(1,1), padding=(0,0)),
    nn.Conv2d(128, 256, kernel_size=(3,3), stride=(2,2), padding=(1,1)), #3
    nn.Conv2d(256, 128, kernel_size=(1,1), stride=(1,1), padding=(0,0)),
    nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(0,0)) #1
])

source = list()
source.append()

################################

import torch;import torch.nn as nn;import torch.nn.functional as F

################################

Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/mnt/hdd1/workspace/torch/.venv/lib/python3.7/site-packages/torch/nn/modules/module.py", line 477, in __call__
    result = self.forward(*input, **kwargs)
  File "/mnt/hdd1/workspace/torch/layer/modules/ssd.py", line 71, in forward
    loc_list.append(j(i).permute(0,2,3,1))
  File "/mnt/hdd1/workspace/torch/.venv/lib/python3.7/site-packages/torch/nn/modules/module.py", line 477, in __call__
    result = self.forward(*input, **kwargs)
  File "/mnt/hdd1/workspace/torch/.venv/lib/python3.7/site-packages/torch/nn/modules/conv.py", line 301, in forward
    self.padding, self.dilation, self.groups)
RuntimeError: Given groups=1, weight of size [16, 128, 3, 3], expected input[1, 256, 1, 1] to have 128 channels, but got 256 channels instead
>>>

################################

Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/mnt/hdd1/workspace/torch/.venv/lib/python3.7/site-packages/torch/nn/modules/module.py", line 477, in __call__
    result = self.forward(*input, **kwargs)
  File "/mnt/hdd1/workspace/torch/layer/modules/ssd.py", line 65, in forward
    x = self.extra[i](x)
  File "/mnt/hdd1/workspace/torch/.venv/lib/python3.7/site-packages/torch/nn/modules/module.py", line 477, in __call__
    result = self.forward(*input, **kwargs)
  File "/mnt/hdd1/workspace/torch/.venv/lib/python3.7/site-packages/torch/nn/modules/conv.py", line 301, in forward
    self.padding, self.dilation, self.groups)
RuntimeError: Given groups=1, weight of size [1024, 1024, 1, 1], expected input[1, 512, 19, 19] to have 1024 channels, but got 512 channels instead

################################

import torch;import torch.nn as nn;import torch.nn.functional as F
from layer.modules.ssd import SSD
a = torch.rand(1,3,300,300)
b = SSD()
c = b(a)

################################

"""
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/mnt/hdd1/workspace/torch/.venv/lib/python3.7/site-packages/torch/nn/modules/module.py", line 477, in __call__
    result = self.forward(*input, **kwargs)
  File "/mnt/hdd1/workspace/torch/layer/modules/ssd.py", line 74, in forward
    loc = torch.cat([t.view(t.shape[0],-1) for t in loc_list],1)
  File "/mnt/hdd1/workspace/torch/layer/modules/ssd.py", line 74, in <listcomp>
    loc = torch.cat([t.view(t.shape[0],-1) for t in loc_list],1)
RuntimeError: invalid argument 2: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Call .contiguous() before .view(). at /pytorch/aten/src/TH/generic/THTensor.cpp:237
"""

################################

import torch;import torch.nn as nn;import torch.nn.functional as F
torch.set_default_tensor_type(torch.cuda.FloatTensor)

################################

a = torch.rand(100)
b = a < 0.5
c = a.masked_fill(b,0)
d = c.sort(descending=True)

################################



################################



################################



################################