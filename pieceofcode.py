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

:

################################