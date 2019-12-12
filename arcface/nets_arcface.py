import torch
import torch.nn as nn
import torch.nn.functional as F





class ArcLoss(nn.Module):
	def __init__(self, feature_dim, cls_dim):

		super().__init__()
		self.W = nn.Parameter(torch.randn(feature_dim, cls_dim))

	def forward(self, X):
		_w = torch.norm(self.W, dim=0)  # (10)
		_x = torch.norm(X, dim=1)  # （N）
		_w = _w.reshape(1, -1)  # 变形，广播(1,10)
		_x =  _x.reshape(-1,1)#变形，广播#(N,1)
		out = torch.matmul(X, self.W)
		cosa = (out/(_x*_w))*0.96#(N,10)#防止梯度爆炸和梯度为0无法更新w，当a=0.cosa=1，不能出现这种情况
		a = torch.acos(cosa)#(N,10)#α不能为0
		top = torch.exp(_x*_w*torch.cos(a + 0.1)+0.001)#防止梯度爆炸和梯度为0无法更新w，exp的指数不能为0
		_top =torch.exp ( _x*_w*torch.cos(a)+0.001)#防止梯度爆炸和梯度为0无法更新w
		bottom = torch.sum(torch.exp(out), dim=1)#（N，）
		bottom = bottom.reshape(-1, 1)
		# return torch.log(top / (bottom - _top + top)+0.002)
		#NLLloss损失里面没有log函数，需要在结果加一个log函数，再用nllloss损失
		return torch.log(top/(bottom - _top + top)+0.002)#防止梯度爆炸和梯度为0无法更新w，log0超级大，需要加一个0.002这些


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
		self.arcloss = ArcLoss(2,10)

	def forward(self, x):

		y1 = self.layers1(x)
		y2 = y1.reshape(y1.size()[0], -1)
		output = self.layers2(y2)
		output1 = self.layers3(output)
		output2 = self.layers4(output1)
		arc_result  = self.arcloss(output1)
		return arc_result,output1