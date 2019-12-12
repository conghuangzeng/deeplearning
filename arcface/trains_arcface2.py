import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
# from nets import Net,CenterLoss
from nets_arcface import Main_net,ArcLoss
from  nets_arcface2 import ArcMarginProduct


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net =Main_net()
arcloss2 = ArcMarginProduct(in_features=2, out_features=10)

# arcloss2  = torch.load()
net = net.to(device)
arcloss2 = arcloss2.to(device)
criterion_cross = nn.CrossEntropyLoss()
criterion_mse = nn.MSELoss()
loss_nll = nn.NLLLoss()
# optim1 = torch.optim.SGD(params=arcloss2.parameters(), lr=0.01)
# optim = torch.optim.SGD(net.parameters(),lr=0.01)
optim1 = torch.optim.Adam(arcloss2.parameters())
optim = torch.optim.Adam(net.parameters())
transform = transforms.Compose([
		# transforms.Resize(28),
		transforms.ToTensor(),
		transforms.Normalize((0.5,), (0.5,)),
	])
dataset = datasets.MNIST("datasets/", train=True, download=True,transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=True)
epoch = 1
losses = []

while True:
	# if epoch>200:
	# 	break
	plt.cla()
	for j, (input, target) in enumerate(dataloader):
		input = input.to(device)


		output1,output2 = net(input)
		target1  = target
		target1 = target1.to(device)


		arc_loss,cosine = arcloss2(output1,target)#改进后的softmax
		print(cosine.shape)
		cosine1 = F.softmax(cosine)
		loss =criterion_cross(arc_loss, target1)#改进后的softmax与标签比较

		optim.zero_grad()
		optim1.zero_grad()
		loss.backward()
		optim.step()
		optim1.step()
		index = torch.argmax(cosine1, dim=1)  # (512,)#output2根本就没用到
		colors =['#A52A2A', '#0000FF', '#FF0000','#7FFFD4','#FFFF00', '#FFC0CB', '#008000', '#00008B','#000000', '#FF00FF']
		# plt.cla()
		# 根据标签来画图
		# plt.ion()
		# plt.clf()  # 清空
		# for i in range(10):
		# 	plt.title('arcloss')
		# 	output1 = output1.cpu()
		# 	output1 = output1.detach()
		# 	cosine = cosine.cpu()
		# 	cosine = cosine.detach()
		# 	# index = index.cpu()
		# 	# target= target.cpu()
		# 	plt.scatter(output1[i ==target , 0], output1[i ==target, 1], color=colors[i], s=2)
		# 	# plt.xlim(left=-10, right=10)
		# 	# plt.ylim(bottom=-10, top=10)
		# 	plt.text(-9, 9, 'epoch={}'.format(epoch))
		# 	plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
		#
		# # plt.show()
		# plt.pause(0.1)
		# plt.ioff()
		if j % 1 == 0:
			losses.append(loss.float())  #
			print("[epochs - {0} - {1}/{2}]loss: {3}".format(epoch, j, len(dataloader), loss.float()))
	torch.save(net, "net_adm_arcface_0927.pth")# 保存网络训练结果

