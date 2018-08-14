import torch
import torch.nn as nn
import torchvision

class SSD(nn.Module):
    def __init__(self):
        super(SSD,self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True).features[:-1]
        for i in vgg
            if isinstance(i, nn.MaxPool2d):
                i.ceil_mode = True

        self.extra = nn.ModuleList([ #19
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

    def forward(self, x):
        #input -> vgg -> extra -> class
        #                      ->

        asdf = list()
        for i in range():
            x = self.vgg[i](x)
            if i in (21, 30):
                asdf.append(x)
        
        for i in range():
            x = self.extra[i](x)
            if i in ():
                asdf.append(x)
