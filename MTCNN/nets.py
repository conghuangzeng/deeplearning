import torch.nn as nn
import torch.nn.functional as F

class Pnet(nn.Module):
    def __init__(self):
        super(Pnet,self).__init__()
        self.layers1=nn.Sequential(
            nn.Conv2d(3,10,3,1,padding=1),#12*12
            nn.BatchNorm2d(10),
            nn.PReLU(),
            nn.MaxPool2d(3,stride=2),#搞成了5*5
            nn.Conv2d(10,16,3,1),#3*3
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 32, 3, 1),#1*1
            nn.BatchNorm2d(32),
            nn.PReLU()
        )
        #P网络的总步长等于每一层的步长相乘，所以反算的时候设置了一个步长为2
        self.sigmoid = nn.Sigmoid()
        self.cond=nn.Conv2d(32,1,1,1)
        self.offset = nn.Conv2d(32,4,1,1)
    def forward(self, x):
        y1=self.layers1(x)
        cond1=self.cond(y1)
        cond = self.sigmoid(cond1)
        offset=self.offset(y1)
        return cond,offset


class Rnet(nn.Module):
    def __init__(self):
        super(Rnet, self).__init__()
        self.layers1 = nn.Sequential(
            nn.Conv2d(3, 28, 3, 1),#22*22
            nn.BatchNorm2d(28),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),#10*10
            nn.Conv2d(28, 48, 3, 1),#8*8
            nn.BatchNorm2d(48),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),#3*3
            nn.Conv2d(48, 64, 2, 1),#2*2
            nn.BatchNorm2d(64),
            nn.PReLU(),

        )
        self.layers2 = nn.Sequential(
            nn.Linear(64 * 2*2, 128),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.PReLU(),


        )
        self.cond = nn.Linear(64, 1)
        self.offset = nn.Linear(64, 4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y1 = self.layers1(x)
        y1 = y1.reshape(y1.size()[0], -1)
        y2 = self.layers2(y1)
        y3 = self.cond(y2)
        cond = self.sigmoid(y3)
        offset = self.offset(y2)

        return cond, offset


class Onet(nn.Module):
    def __init__(self):
        super(Onet, self).__init__()
        self.layers1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),#46*46
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),#22*22
            nn.Conv2d(32, 64, 3, 1, padding=0),#20*20
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),#9*9
            nn.Conv2d(64, 64, 3, 1, padding=0),#7*7
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(2),#3*3
            nn.Conv2d(64, 128, 2, 1),#2*2
            nn.BatchNorm2d(128)
        )
        self.layers2 = nn.Sequential(
            nn.Linear(128 * 2*2, 128),
            nn.PReLU(),
            nn.Linear(128 , 64),
            nn.PReLU()


        )
        self.cond = nn.Linear(64, 1)
        self.offset = nn.Linear(64, 4)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y1 = self.layers1(x)
        y1 = y1.reshape(y1.size()[0], -1)
        y2 = self.layers2(y1)
        y3 = self.cond(y2)
        cond = self.sigmoid(y3)
        offset = self.offset(y2)
        return cond, offset
import torch
if __name__ == '__main__':

    pnet  = Pnet()
    rnet = Rnet()
    onet  = Onet()

    x = torch.randn(2,3,300,600)
    y = pnet(x)[0]
    print(y.shape)
