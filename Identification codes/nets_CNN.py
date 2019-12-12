# import os
# import numpy as np
import torch
import torch.nn as nn
# import torch.utils.data as data
# import Sampling_train
#
# BATCH = 32
# EPOCH = 100

# save_path = "net.pth"
#
# if not os.path.exists(save_path):
#     os.makedirs(save_path)

class Mynet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers1  = nn.Sequential(
            nn.Conv2d(3,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d( 32, 64,3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32,3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layers2 = nn.Linear(32*7*30,40)
    def forward(self, x):
       y1= self.layers1(x)
       # print(y1.shape)
       y2 = y1.reshape(y1.size()[0],-1)
       out = self.layers2(y2).reshape(-1,4,10)
       # out = torch.zeros(out.size()[1],10).scatter_(1,)
       return out



if __name__ == '__main__':

    net  =Mynet()
    x =torch.randn(2,3,60,240)
    output =net(x)
    print(output.shape)










