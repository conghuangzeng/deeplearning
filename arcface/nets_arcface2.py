from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn import Parameter
import torch.nn as nn
import math


class ArcMarginProduct(nn.Module):
    def __init__(self,in_features,out_features,s=30.0,m=0.50,easy_margin=False):
        super(ArcMarginProduct,self).__init__()
        self.in_features = in_features#前面4行需要先把init里面的变量全部定义了，不然把网络放到优化器里面，会有报警，说这个类的网络参数是一个空的list
        self.out_features = out_features
        self.s = s
        self.m = m
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)#给w赋值
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)#常数
        self.sin_m = math.sin(m)#常数
        self.th = math.cos(math.pi - m)#cos(math.pi - m)=-cos(m)，就是一个常数
        self.mm = math.sin(math.pi - m) * m#就是一个常数


    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        input  = input.to(self.device)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.001 - torch.pow(cosine, 2)).clamp(0, 1))#1.000001也是为了防止梯度爆炸和梯度消失，防止cos = ±1的情况
        phi = cosine * self.cos_m - sine * self.sin_m#phi=cos（a+m），cosine=cosa
        label = label.to(self.device)
        if self.easy_margin:#感觉phi的取值可以简化
            phi = torch.where(cosine > 0, phi, cosine)#这句有用，相当于是取a是0-90度，取cos（a+m）a是90-180度，取cosa，当phi取cosa的时候，output固定 = 30
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        label = label.cpu().long()
        one_hot = torch.zeros(label.size(0), 10).scatter_(1, label.view(-1, 1), 1).to(self.device)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4#w的模为1，x的模为1.相当于exp也不要了，只要cos（a+m）和cosa，原公式理解为cos（a+m）除以（cos（a+m）+cosa-onehot对应的cosa）
        output *= self.s#output太小了，乘以一个常数，30
        # print(output)

        return output,cosine#cosine为w*x的两个特征点，N，2的值。

class Main_net(nn.Module):

	def __init__(self):
		super(Main_net, self).__init__()
		self.layers1 = nn.Sequential(
			nn.Conv2d(1, 32, 3, 1),  # 26*26
			nn.ReLU(),
			nn.MaxPool2d(2),  # 13*13
			nn.Conv2d(32, 64, 3, 1),  # (11*11)
			nn.ReLU(),
			nn.MaxPool2d(2),  # 5*5
			nn.Conv2d(64, 16, 3, 1),  # 3*3
			nn.ReLU()
					)

		self.layers2 = nn.Sequential(
			nn.Linear(in_features=16 * 3 * 3, out_features=64, bias=True),
			nn.ReLU(),
			)
		self.layers3 = nn.Linear(in_features=64, out_features=2, bias=True)  #
		self.layers4 = nn.Linear(in_features=2, out_features=10, bias=True)  #
		self.arcloss = ArcMarginProduct(2,10)

	def forward(self, x):

		y1 = self.layers1(x)
		y2 = y1.reshape(y1.size()[0], -1)
		output = self.layers2(y2)
		output1 = self.layers3(output)
		output2 = self.layers4(output1)
		arc_result  = self.arcloss(output1)
		return arc_result,output1
