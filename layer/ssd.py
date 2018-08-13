import torch
import torchvision

class SSD(torch.nn.Module):
    def __init__(self):
        self.vgg = torchvision.models.vgg16(pretrained=True)

        self.
