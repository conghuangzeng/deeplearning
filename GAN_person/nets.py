import torch
import torch.nn as nn
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os

class D_Net(nn.Module):
    def __init__(self):
        super(D_Net,self).__init__()
        self.dnet = nn.Sequential(
            #原始数据96*96
            #所有偏执设为Flase
            nn.Conv2d(3, 64,5, 3,1,bias=False),  # 32*32
            #D网络的第一层不用BN层
            # nn.BatchNorm2d(256),#
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(64, 128, 4, 2,1,bias=False),  # 16*16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(128,256,4,2,1,bias=False),#8*8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(256,512,4,2,1,bias=False),#4*4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0,bias=False),#1*1
            nn.Sigmoid()
        )

    def forward(self, x):
        y =self.dnet(x)
        return y

class G_Net(nn.Module):
    def __init__(self):
        super(G_Net,self).__init__()
        self.gnet = nn.Sequential(
            #原始是1*1,全部采用偶数的卷积核，偏置全部设为Flase，还有relu的参数设为True
            nn.ConvTranspose2d(256,512, 4, 1,0,bias=False),  #4*4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256,4, 2,1 ,bias=False),  # 8*8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2,1,bias=False),  # 16*16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1,bias=False),  # 32*32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,3, 5, 3, 1,bias=False), # 96*96
            nn.Tanh()
        )
    def forward(self, x):
        y =self.gnet(x)
        return y

if __name__ == '__main__':
    dnet = D_Net()
    gnet = G_Net()
    x = torch.randn(1,3,96,96)
    y = dnet(x)
    print(y.shape)
    x1 = torch.randn(1, 256, 1,1)
    y1 = gnet(x1)
    print(y1.shape)



