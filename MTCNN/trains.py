import torch
import  numpy as np
import torch.utils.data as data
from torchvision import  transforms,datasets
import torch.nn as nn
from net_datas import  Facedataset
from nets import Rnet,Onet,Pnet
pnet = Pnet()
rnet = Rnet()
onet = Onet()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])
sigmoid = nn.Sigmoid()
import nets
class Trainer:
    def __init__(self,net,net_path_old,net_path_new,dataset_path):
        self.net = net
        self.net_path_old = net_path_old
        self.net_path_new = net_path_new
        self.dataset_path = dataset_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def trains(self):
        my_data = Facedataset(self.dataset_path,transform=transform)
        train_data =data.DataLoader(dataset=my_data, batch_size=300, shuffle=True,num_workers=2)

        lossfun_1 = nn.MSELoss()
        lossfun_2 = nn.BCELoss()
        self.net = torch.load(self.net_path_old)
        self.net = self.net.to(self.device)
        self.net.train()
        optim = torch.optim.Adam(self.net.parameters(),lr=0.001)
        epoch = 1
        loss_average_total_now = 0
        while True:
            if epoch>15:
                break
            loss_average_total_before = loss_average_total_now
            loss_total_now = 0
            for i,(img_data,label_cond,label_offset) in enumerate(train_data):
                img_data = img_data.to(self.device)
                label_cond, label_offset = label_cond.to(self.device), label_offset.to(self.device)
                output=self.net(img_data)
                output_cond,output_offset =output
                output_cond = output_cond.view(-1,1)#(N,1)
                output_cond = output_cond.squeeze(1)#(N)
                output_offset = output_offset.reshape(-1,4)#(N,4)


                #部分样本不参与置信度损失的计算
                index = torch.lt(label_cond,2)
                _label_cond = torch.masked_select(label_cond,index)#（N,）1维
                _output_cond = torch.masked_select(output_cond,index)
                loss_cond = lossfun_2(_output_cond, _label_cond)

                #负样本不参与偏移量的计算
                label_cond = label_cond.view(-1,1)#（N,1）
                index1 = torch.gt(label_cond,0).squeeze(1)#（N,1）降维成N
                # index1 = torch.gt(label_cond, 0)#这一句就可以替代上面的两句
                _label_offset = label_offset[index1]#(N,4)
                _output_offset = output_offset[index1]#(N,4)
                loss_offset =lossfun_1(_output_offset,_label_offset)

                loss_total = loss_cond+loss_offset
                optim.zero_grad()
                loss_total.backward()
                optim.step()
                loss_total_now  = loss_total.item()+loss_total_now
                loss_average_total_now = (loss_total_now) / (i+1)
                if i %10== 0:
                    print("训练网络：{0}，训练轮次：{1}，训练批次：{2}，单次损失：{3}，平均损失：{4}".format(self.net_path_new[0:4],epoch,i//10,loss_total.item(),loss_average_total_now))
                    print("condloss:{0},offset:{1}".format(loss_cond.item(),loss_offset.item()))
            print("上次平均损失：{0}  本次平均损失:{1}".format(loss_average_total_before,loss_average_total_now))
            epoch += 1

            # if loss_average_total_now<loss_average_total_before:
            #     torch.save(self.net,self.net_path)
            torch.save(self.net,self.net_path_new)
            print("保存成功")
        print("训练完成")
