import torch
from torchvision.models._utils import IntermediateLayerGetter
import torch.nn as nn
import torchvision.models as models

RES18 = models.resnet18(pretrained=True)
# for name,module in RES18.named_children():
#     print(name,module)

class Fcn16(nn.Module):
    def __init__(self,n_class=34):
        super(Fcn16,self).__init__()
        # pretrain RseNet
        self.res = IntermediateLayerGetter(RES18,{'layer3':'feat1','layer4':'feat2'}) # 1/16, 1/32

        # avgpool
        self.avgpool = nn.AvgPool2d(7,1,3)

        # 1*1 convolution
        self.cov1 = nn.Conv2d(256,n_class+1,1)
        self.cov2 = nn.Conv2d(512, n_class + 1, 1)

        # deconvolution
        self.decov2 = nn.ConvTranspose2d(n_class+1,n_class+1,4,2)
        self.decov16 = nn.ConvTranspose2d(n_class+1,n_class+1,32,16)



    def forward(self,x):
        h=x
        res = self.res(h)
        _,h32 = res.popitem()
        _,h16 = res.popitem()
        h16 = self.cov1(h16)  # 1/16
        h32 = self.cov2(h32)  # 1/32
        h32 = self.decov2(h32)  # 1/16
        h32 = h32[:,:,1:1+h16.size(2),1:1+h16.size(3)]
        h=h16+h32
        h = self.decov16(h)
        # print(h.size())
        # _,_,Hx,Wx = x.size()
        # _,_,Hh,Wh=h.size()
        # offset = int((Hh-Hx)/2 -1)
        h=h[:,:,7:7+x.size(2),7:7+x.size(3)]
        return h



# net = Fcn16()
# # res = IntermediateLayerGetter(RES18,{'layer4':'feat1'})
# out = net(torch.rand(1,3,1024,4096))
# print(out.size())
