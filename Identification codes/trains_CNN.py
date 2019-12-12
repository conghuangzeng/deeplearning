import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch
from torchvision import  transforms
from  nets_CNN import Mynet
from nets_data_CNN import Sampling
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# net = Mynet().to(device)
# net = torch.load("net.pth").to(device)#损失0.0.25效果不好
# net = torch.load("net_shuangxiang.pth")#双向RNN效果不好
net = torch.load("net_CNN.pth").to(device)
optim = torch.optim.Adam(net.parameters())
loss_mse_fun = nn.MSELoss()
loss_cross_fun = nn.CrossEntropyLoss()
batch_size = 100
my_data = Sampling(r"train_img")
train_data = data.DataLoader(my_data,batch_size=batch_size,shuffle=True)
softmax = nn.Softmax()
#根据API在实例化的时候就要把dim放进去。
#    def __init__(self, dim=None):
#         super(Softmax, self).__init__()
#         self.dim = dim
epoch  = 1
while True:
    if epoch>100:
        break
    loss_total = 0
    for i ,(img_data,label) in enumerate(train_data):
        img_data,label = img_data.to(device),label.to(device)
        output = net(img_data)
        # output1 = output
        # print(output.shape)#(N,4,10)
        # print(label.shape)#(N,4,10)
        # output = output[:,]
        output1= F.softmax(output,dim=2)

        # output1 = softmax(output, dim=2)
        label = label.float()
        # print(label)
        # label1 = torch.zeros(label.size()[0],10).scatter_(1,label.view(-1,1),1)
        loss = loss_mse_fun(output1,label)

        optim.zero_grad()
        loss.backward()
        optim.step()
        loss_total+=  loss.item()
        if i%5 == 0:
        #
        #
        #     output2 = torch.argmax(output1,2).cpu().detach().numpy()
        #     label2 = torch.argmax(label,2).cpu().detach().numpy()
        #     # print(output)#(N,4)
        #     # print(label)#(N,4)
        #     # print(np.sum(output==label))
        #     acc = np.sum(output2==label2,dtype=np.float32)/(batch_size*4)
        #     print("epoch:{},i:{},loss:{},acc:{}%".format(epoch,i,loss.item(),acc*100))
            # print("label:",y[0])
            # print("output:",output[0])
        # total = 0
        # correct1 = 0
        # correct2 = 0
        # if i % 10 == 0:
        #     total += img_data.size()[0]
        #     #
        #     #
        #     correct1 += (y1 == output1.data).sum()
        #     acc1 = correct1.float() / total
        #     correct2 += (output2.data == y2).sum()
        #     acc2 = correct2.float() / total
            print("epoch:{},i:{},loss:{}".format(epoch, i, loss_total/(i+1)))
    epoch+= 1
    torch.save(net,"net_CNN.pth")
    print("保存路径成功")