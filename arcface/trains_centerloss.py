import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from nets_centerloss import Main_net,CenterLoss
import os

# def centerloss_fun(output1,index):
# 	centerloss1 = 0
# 	for i in range(10):
# 		x = output1[i==index,0]
# 		y = output1[i==index,1]
# 		x_average = torch.mean(x)
# 		y_average = torch.mean(y)
# 		distance = torch.sum((x_average - x) ** 2 + (y_average - y) ** 2)
# 		centerloss1+= distance/2
# 	return centerloss1#这个用矩阵进行改良
net_path =  "net1_center.pth"
if os.path.exists(net_path):
	net = torch.load(net_path)
else:
	net = Main_net()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)
criterion_cross = nn.CrossEntropyLoss()
optim = torch.optim.Adam(net.parameters(),lr=0.001)
transform = transforms.Compose([
		transforms.Resize(28),
		transforms.ToTensor(),
		transforms.Normalize((0.5,), (0.5,)),
	])
dataset = datasets.MNIST("datasets/", train=True, download=True,transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=True)
testdataset = datasets.MNIST("datasets/", train=False, download=True,transform=transform)
testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1000, shuffle=False)

epoch = 1
losses = []

while True:
	# if epoch>200:
	# 	break
	# plt.cla()
	for j, (input, target) in enumerate(dataloader):
		input = input.to(device)
		output1,output2 = net(input)
		target = target.to(device)
		loss = net.get_loss(output1,output2, target)
		index = torch.argmax(output2, dim=1)#(512,)
		optim.zero_grad()
		loss.backward()
		optim.step()
		colors =['#A52A2A', '#0000FF', '#FF0000','#7FFFD4','#FFFF00', '#FFC0CB', '#008000', '#00008B','#000000', '#FF00FF']
		# for i in range(10):
		# 	x, y = output1[index == i, 0], output1[index == i, 1]
		# 	x, y = x.cpu().detach(), y.cpu().detach()
		# 	x, y = x.numpy(), y.numpy()
		# 	plt.scatter(x, y, c=colors[i], s=5)
		# 	plt.legend(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], loc=1)
		# 	plt.rcParams['font.sans-serif'] = ['SimHei']
		# 	plt.rcParams['axes.unicode_minus'] = False
		# 	# plt.title('全连接无centerloss',color = "red")
		# plt.title("卷积+交叉熵", color="green")
		# # plt.show()
		# plt.pause(0.1)
		# plt.cla()
		#根据标签来画图
		plt.ion()
		plt.clf()  # 清空
		for i in range(10):
			plt.title('Centerloss')
			output1 = output1.cpu()
			output1 = output1.detach()
			# index = index.cpu()
			# target= target.cpu()
			plt.scatter(output1[i == target, 0], output1[i == target, 1], color=colors[i],s=2)
			# plt.xlim(left=-10, right=10)
			# plt.ylim(bottom=-10, top=10)
			plt.text(-9, 9, 'epoch={}'.format(epoch))
			plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')

		# plt.show()
		plt.pause(0.1)
		plt.ioff()
		if j % 1 == 0:
			losses.append(loss.float())  #
			print("[epochs - {0} - {1}/{2}]loss: {3}".format(epoch, j, len(dataloader), loss.float()))
	epoch += 1
		#下面的这一串理解清楚。
	# torch.save(net, "net1_center.pth")  # 保存网络训练结果
	# torch.save(centerloss, "net1_centerloss.pth")


