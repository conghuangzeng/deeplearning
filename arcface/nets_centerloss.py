import torch
import torch.nn as nn
import torch.nn.functional as F
class Main_net(nn.Module):

	def __init__(self):
		super(Main_net,self).__init__()
		self.layers1 = nn.Sequential(
			nn.Conv2d(1,32,3,1),#26*26
			nn.ReLU(),
			nn.MaxPool2d(2),#13*13
			nn.Conv2d(32, 64, 3, 1),#(11*11)
			nn.ReLU(),
			nn.MaxPool2d(2),#5*5
			nn.Conv2d(64, 16, 3, 1),#3*3
			nn.ReLU()
		)

		self.layers2 = nn.Sequential(
			nn.Linear(in_features=16*3*3, out_features=64, bias=True),
			nn.ReLU(inplace=False),
		)
		self.layers3 = nn.Linear(in_features=64, out_features=2, bias=True)#
		self.layers4 = nn.Linear(in_features=2, out_features=10, bias=True)#
		self.softmax = nn.Softmax()
		self.centerloss = CenterLoss(cls_num=10, feature_num=2)#将centerloss实例化
		self.crossentropyLoss = nn.CrossEntropyLoss()
	def forward(self, x):
		y1 = self.layers1(x)
		y2  = y1.reshape(y1.size()[0],-1)
		output = self.layers2(y2)
		output1 = self.layers3(output)
		output21 = self.layers4(output1)
		output2 =self.softmax(output21)
		return output1,output2
	def get_loss(self,output1,output2,label,):
		centerloss = self.centerloss(output1,label)
		loss1 = self.crossentropyLoss(output2,label)
		loss_tatol = centerloss*0.01+loss1
		return loss_tatol



class CenterLoss(nn.Module):

	def __init__(self,cls_num,feature_num):
		super(CenterLoss,self).__init__()
		self.cls_num = cls_num
		self.center = nn.Parameter(torch.randn(cls_num,feature_num))

	def forward(self, xs,ys):
		# xs = F.normalize(xs).float()#这一步不需要
		# center_exp = self.center.index_select(dim=0, index=ys.long()).float()#(N,2)
		center_exp  = self.center[ys.long()].float()#花式索引也ok
		count = torch.histc(ys, bins=self.cls_num, min=0, max=self.cls_num-1).float()#求出每个类别的个数
		# count_dis = count.index_select(dim=0, index=ys.long()).float()
		count_dis = count[ys.long()].float()#花式索引也ok

		#原本是每个类别其他点到中心点的欧氏距离，求和除以每个类别的总个数，搞成求一个距离除以一个count_dis，count_dis是每一个对应的所在类别的总个数
		centerloss1 = torch.sum(torch.sqrt(torch.sum((xs-center_exp)**2,dim=1))/count_dis)#求欧式距离，一个值

		return centerloss1
