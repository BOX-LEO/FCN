import torch
from torchvision.models._utils import IntermediateLayerGetter
import torch.nn as nn
import torchvision.models as models

RES18 = models.resnet18(pretrained=True)
# for name,module in RES18.named_children():
#     print(name,module)

class Fcn32(nn.Module):
    def __init__(self,n_class=34):
        super(Fcn32,self).__init__()
        # pretrain RseNet
        self.resl4 = IntermediateLayerGetter(RES18,{'layer4':'feat1'}) # 1/32

        # avgpool
        self.avgpool = nn.AvgPool2d(7,1,3)

        # 1*1 convolution
        self.cov1 = nn.Conv2d(512,n_class+1,1)

        # deconvolution
        self.decov = nn.ConvTranspose2d(n_class+1,n_class+1,64,32)



    def forward(self,x):
        h=x
        k,h = self.resl4(h).popitem()
        h= self.avgpool(h)
        h = self.cov1(h)
        h = self.decov(h)
        # _,_,Hx,Wx = x.size()
        # _,_,Hh,Wh=h.size()
        # offset = int((Hh-Hx)/2 -1)
        h=h[:,:,15:15+x.size(2),15:15+x.size(3)]
        return h



# net = FCN32()
# # res = IntermediateLayerGetter(RES18,{'layer4':'feat1'})
# out = net(torch.rand(1,3,4096,2048))
# print(out.size())
