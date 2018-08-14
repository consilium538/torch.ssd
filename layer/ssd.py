import torch
import torch.nn as nn
import torchvision

f_map = ((38, 512, 4),
        (19, 1024, 6),
        (10, 512, 6),
        (5, 256, 6)
        (3, 256, 6)
        (1, 128, 4))
num_classes = 21

class SSD(nn.Module):
    def __init__(self):
        super(SSD,self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True).features[:-1]
        for i in vgg.parameters(): #freeze pretrained network
            i.requires_grad = False
        for i in vgg:
            if isinstance(i, nn.MaxPool2d): # make dimension correct
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

        self.mbox_loc = nn.ModuleList(
                [nn.Conv2d(x[0],x[1]*4,kernel_size=3,padding=1) for x in f_map])
        self.mbox_conf = nn.ModuleList(
                [nn.Conv2d(x[0],x[1]*num_classes,kernel_size=3,padding=1) for x in f_map])

    def forward(self, x):
        #input -> vgg -> extra -> class
        #                      ->

        feature_list = list()
        loc_list = list()
        conf_list = list()

        for i in range():
            x = self.vgg[i](x)
            if i in (21):
                feature_list.append(x)
        
        for i in range():
            x = self.extra[i](x)
            if i % 2 == 1:
                feature_list.append(x)

        for (i,j,k) in zip(feature_list, self.mbox_loc, self.mbox_conf):
            loc_list.append(j(i).permute(0,2,3,1))
            conf_list.append(k(i).permute(0,2,3,1))

        loc = torch.cat([t.view(t.size(0),-1) for in loc_list],1)
        conf = torch.cat([t.view(t.size(0),-1) for in conf_list],1)

        output = (
                loc.view(loc.size(0),-1,4),
                conf.view(conv.size(0),-1,num_classes),
        )
