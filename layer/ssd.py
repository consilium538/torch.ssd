import torch
import torch.nn as nn
import torchvision

class SSD(torch.nn.Module):
    def __init__(self):
        super(SSD,self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True).features

        self.extra = nn.Sequential(
            nn.Conv2d(),
        )

    def forward(self, x):
        #input -> vgg -> extra -> class
        #                      ->
        
        for i in range():
            x = self.vgg[i](x)
            if i in (21, 30):
                asdf.append(x)
        
        for i in range():
            x = self.extra[i](x)
            if i in ():
                asdf.append(x)