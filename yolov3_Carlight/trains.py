import torch
import torch.utils.data as data
import os
import torch.nn as nn
from  torchvision import transforms
import PIL.Image as pimg
import numpy as np
import math
from Nets import Main_net
from  Nets_data import Yolodataset
import torch.nn.functional as F

transform = transforms.Compose([
	transforms.Resize((416,416)),
	transforms.ToTensor(),
	transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net_path = r"net1108.pth"
if os.path.exists(net_path):
    net = torch.load(net_path)
    print("加载网络成功")
else:
    net = Main_net()
    net = net.to(device)
net.train()#必不可少的，理解

loss_mse_fun  = nn.MSELoss()
loss_cross_fun  = nn.CrossEntropyLoss()
sigmoid = nn.Sigmoid()
loss_bce_fun = nn.BCELoss()
loss_nlll_fun  = nn.NLLLoss()
softmax = nn.Softmax(dim=1)

class Trainer:

#如何设置损失函数，非常重要
    def loss_fun(self, output, label, alpha):
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(output.size()[0], output.size()[1], output.size()[2], 3, -1)
        mask_object = label[..., 0] > 0  # 这里为什么要设为0,大于0 的认为是正样本
        mask_no_object = label[..., 0] == 0
        label = label.double()  # 转换为double张量，float张量会有报警
        output = output.double()
        # 损失的设计重点掌握
        #另外一种损失
        loss_object_iou = loss_mse_fun(output[mask_object][:, 0:1],label[mask_object][:, 0:1])  # 平方求和试试
        loss_object_offset_xy_wh = loss_mse_fun(output[mask_object][:, 1:5],label[mask_object][:, 1:5])
        loss_no_object = loss_mse_fun(output[mask_no_object], label[mask_no_object])
        output_class = output[mask_object][:, 5:]
        label_class = label[mask_object][:, 5:]
        loss_object_class = loss_mse_fun(softmax(output_class), label_class)
        loss_total = (loss_object_offset_xy_wh  + loss_object_iou + loss_object_class) * alpha + loss_no_object * ( 1 - alpha)
        return loss_total
    def train(self):
        yolodataset = Yolodataset(r"dataset_data", transform=transform)

        batch_size =3#训练多张图片，批次千万不能给1，重点
        train_data = data.DataLoader(dataset=yolodataset, batch_size=batch_size, shuffle=True)
        loss_average_before =40.0#类别损失乘以10：当前损失0.033，#r"net1006_num15_sigmoid.pth"还可以。
        # 类别损失不乘以10：当前损失0.05，r"net1007_num15_sigmoid.pth"分类分不出来。
        epoch = 1
        while True:
            decay_rate = 0.05


            lr  = 0.001/(1+decay_rate*epoch)
            optim = torch.optim.Adam(net.parameters(), lr=lr)
# 	break

            loss_total = 0

            for i ,label  in enumerate(train_data):
                img_data,detetion_label_13, detetion_label_26, detetion_label_52 = label

                img_data, detetion_label_13, detetion_label_26, detetion_label_52  =img_data.to(device), detetion_label_13.to(device), detetion_label_26.to(device), detetion_label_52.to(device)
                # print(img_data.size())#(1,C,H,W)数据量太大，一张一张的传
                output = net(img_data)
                # print(output)
                detetion_out_13, detetion_out_26, detetion_out_52 = output
                # print(detetion_out_13.size())
                # print(detetion_label_13.size())#
                # print(torch.nonzero(detetion_label_13[...,0]>0.5))
                loss_13 = self.loss_fun(detetion_out_13,detetion_label_13,0.8)
                loss_26 = self.loss_fun(detetion_out_26, detetion_label_26, 0.8)
                loss_52 = self.loss_fun(detetion_out_52, detetion_label_52, 0.8)

                loss = loss_13+loss_26+loss_52

                optim.zero_grad()
                loss.backward()
                optim.step()

                loss_total+=loss
            lenth =os.listdir(r"dataset_data")

            print("训练轮次：{0}，训练批次：{1}，损失：{2}，平均损失：{3}".format(epoch,i+1,loss_total.item(),loss_total.item()/(i+1)))
            # torch.save(net.state_dict(), "{0}/{1}".format(params_path,"ckpt0930.pth"))#保存参数
            loss_average_now = loss_total.item() / (i + 1)
            # print("当前平均损失:{0}".format(loss_average_before))
            # if loss_average_now<loss_average_before:
            if epoch%10==0:
                torch.save(net, r"net1109.pth")
                print("网络保存成功")
                loss_average_before = loss_average_now
                print("当前平均损失:{0}".format(loss_average_before))
            epoch+=1

if __name__ == '__main__':
    trainer =Trainer()
    trainer.train()



