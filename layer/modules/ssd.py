import torch
import torch.nn as nn
import torchvision
import numpy as np
from itertools import product
from math import sqrt

# pytorch conv2 format : (N, C, H, W), vgg pretrained color : (R, G, B)
# normalized with mean = [0.485, 0.456, 0,406], std = [0.299, 0.224, 0.225]

f_map = ((38, 512, 4, (2,)),
        (19, 1024, 6, (2, 3)),
        (10, 512, 6, (2, 3)),
        (5, 256, 6, (2, 3)),
        (3, 256, 4, (2,)),
        (1, 256, 4, (2,)))
num_classes = 21

class SSD(nn.Module):
    def __init__(self):
        """
        Single Shot multibox Detector layer
        in
        """
        super(SSD,self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True).features[:-1]
        for i in self.vgg.parameters(): #freeze pretrained network
            i.requires_grad = False
        for i in self.vgg:
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
                [nn.Conv2d(x[1],x[2]*4,kernel_size=3,padding=1) for x in f_map])
        self.mbox_conf = nn.ModuleList(
                [nn.Conv2d(x[1],x[2]*num_classes,kernel_size=3,padding=1) for x in f_map])
        self.default_box = self._defaultbox()

    def forward(self, x):
        #input -> vgg -> extra -> class
        #                      ->

        feature_list = list()
        loc_list = list()
        conf_list = list()

        for i in range(29):
            x = self.vgg[i](x)
            if i in (21,):
                feature_list.append(x)

        for i in range(10):
            x = self.extra[i](x)
            if i % 2 == 1:
                feature_list.append(x)

        for (i,j,k) in zip(feature_list, self.mbox_loc, self.mbox_conf):
            loc_list.append(j(i).permute(0,2,3,1).contiguous())
            conf_list.append(k(i).permute(0,2,3,1).contiguous())

        loc = torch.cat([t.view(t.shape[0],-1) for t in loc_list],1)
        conf = torch.cat([t.view(t.shape[0],-1) for t in conf_list],1)

        output = (
                loc.view(loc.size(0),-1,4),
                conf.view(conf.size(0),-1,num_classes),
                torch.cuda.FloatTensor(self.default_box).view(-1,4)
        )

        return output

    def _defaultbox(self):
        """
        default box dimention
        output = ( num_defualt, ( cx,cy,w,h ) )
        """

        #num_default = reduce(lambda x,y:x+y, (x[0]*x[0]*x[2] for x in f_map))
        box_size = np.linspace(0.2, 0.9, 6) # 가장 큰 defualt box의 s_k`의 경우 scale 계산 어찌?
        box_prime_size = np.linspace(0.34, 1.04, 6)
        box_prime_size = np.sqrt(box_size,box_prime_size)
        default_box = list()
        for k, f in enumerate(f_map):
            for i, j in product(range(f[0]), repeat=2):
                cx, cy = (i+0.5)/f[0] , (j+0.5)/f[0]

                s_k = box_size[k]
                default_box.append((cx,cy,s_k,s_k))

                s_k_p = box_prime_size[k]
                default_box.append((cx,cy,s_k_p,s_k_p))
                for r in f[3]:
                    default_box.append((cx,cy,s_k*sqrt(r),s_k/sqrt(r)))
                    default_box.append((cx,cy,s_k/sqrt(r),s_k*sqrt(r)))

        return default_box
