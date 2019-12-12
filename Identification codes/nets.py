import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
# import Sampling_train
#
# BATCH = 32
# EPOCH = 100

# save_path = "net.pth"
#
# if not os.path.exists(save_path):
#     os.makedirs(save_path)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(180,256),
            nn.BatchNorm1d(256),
            nn.ReLU())
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=4,
            batch_first=True,
        bidirectional=False)
    def forward(self, x):
        #[32,3,60,240]--[32,180,240]--[32,240,180]
        x = x.reshape(-1,180,240).permute(0,2,1)
        #[32,240,180]--[32*240,180]
        x = x.reshape(-1,180)
        #[32 * 240, 180]--[32*240,256]
        fc =self.fc(x)
        #[32*240,256]--[32,240,256]
        fc = fc.reshape(-1,240,256)
        #[-1,240,256]--[-1,240,128]
        lstm,(h_n,h_c) = self.lstm(fc,None)#LSTM值得是h1+h2+h3++++hn所有层的输出相加
        #h_n,h_c指的就是当前层的输出和当前层的细胞状态
        #[-1, 240, 128]--[-1,128]
        out = lstm[:,-1,:]
        return out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=4,
            batch_first=True,bidirectional=False
        )
        self.fc = nn.Linear(256,10)

    def forward(self, x):
        #[32,128]--[32,1,128]
        x = x.reshape(-1,1,128)
        #[32,1,128]--[32,4,128]
        # x = x.expand(-1,4,128)
        x = x.repeat(1,4,1)#这个也是一样的效果
        # print(x.shape)
        #[32,4,128]--[32,4,256]
        lstm,(h_n,h_c) = self.lstm(x,None)
        #[32,4,256]--[32*4,256]
        y1 = lstm.reshape(-1,256)
        #[32 * 4, 256]--[32*4,10]
        out = self.fc(y1)
        #[32*4,10]--[32,4,10]
        output = out.reshape(-1,4,10)
        # output = torch.softmax(out,2)

        return output

class SEQ2SEQ(nn.Module):
    def __init__(self):
        super(SEQ2SEQ,self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)

        return decoder

if __name__ == '__main__':

    net  = SEQ2SEQ()
    x =torch.randn(2,3,60,240)
    output =net(x)
    print(output.shape)










