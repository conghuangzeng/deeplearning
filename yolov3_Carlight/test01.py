dataset =[]
import numpy as np
# a = open(r"data-txt/label.txt",mode="r").readlines()
# print(a)
# dataset.extend(a)
# print(dataset)
# b =a[0].strip().split()
# # print(b)
# line = a[0].strip().split()
# print(line)
# import math
#
# a,b = math.modf(7/6)
# print(a)
# print(b)
# d = []
# def one_hot(cls_num, v):
#     b = np.zeros(shape=(cls_num))
#     b[v] = 1.
#     return b
# c = one_hot(10,9)
# print(c)
# d = [1,*c]
# print(d)
import torch
# a = torch.randn(6,2)
# b = torch.randn(6,3)
# c = torch.cat((a,b),dim=1)
# print(c)
# print(c.size())
# anchor_groups = {
# 	13:[[1,1],[2,2],[3,3]],
# 	26:[[4,4],[5,5],[6,6]],
# 	52:[[7,7],[8,8],[9,9]]}
# anchor_groups1 = torch.FloatTensor(anchor_groups[13])
# print(anchor_groups1.size())
# a = anchor_groups1
# a = torch.randn(1,2,2,3,15)
# # # print(a)
# index = a[...,0]>0.1
# print(index.size())
# index1 = index.nonzero()
# # print(index1)
# print(index1.size())
# b = a[index]
# # print(b)
# print(b.size())
# ff = b[:,3]*anchor_groups1[:,0]
# print(ff.size())
#
# #
# print(a)
# import torch
# a = torch.arange(1,5)
# b = torch.arange(1,5)
# c = torch.arange(1,5)
# d = torch.arange(1,5)
# e= torch.arange(1,5)
# f = torch.stack((a,b,c,d,e),dim=0)
# # f = torch.cat((a,b),dim=-1)
# print(f)
# print(f.shape)

# import torch
# a = torch.randn(1,13,13,3,15)
# index = a[...,0]>0.3
# # print(index)
# print(index.size())
# b = a [index][0]
# print(a[index][0])
# print(b.size())
# lable_txt = open(r"dataset_data/label.txt").readlines()
# # lable_txt = lable_txt.split()
# print(len(lable_txt))
# a = lable_txt[0]
# # b = a.strip().split()
# # print(b)
# import torch
# import utils
# a = torch.arange(1,31).reshape(2,15)
# a1 = a[:,0:5]
# # a1 = a1.long()
# # # print(a.size())
# a_cls = a[:,5:15]
# print(a_cls.size())
# b = torch.argmax(a_cls,dim=1)
# # print(b)
# b1 = b.reshape(-1,1)
# a2 = torch.cat((a1,b1),dim=1)
# print(a2.size())
# print(a2)
# lis = []
# lis_tatal =[]
# a2 = a2.numpy()
# # for i,a3 in enumerate(a2):
# # 	print(a3.shape)
# 	# for j in  range(10):
# 	# if int(a3[5])==5:
# 	# 	lis.append(a3)
# 	# print(lis)
# index=np.where(a2[:,5]==9)
# print(index)
# a3 = a2[index]
# print(a3)
	# print(a4.shape)
		# 	# print(a3[0])
	# if len(lis)==0:
	# 	boxes = np.array([])
	# 	if len(lis) == 1:
	# 		boxes = np.array(lis)
	# 	lis_tatal.append(boxes)
	# if len(lis) > 1:
# 	# 	boxes1 = np.stack(lis,axis=0)
# nms_boxes = utils.nms(a3[0:5],i = 0.1,isMin=False)
# 	# 	lis_tatal.append(nms_boxes)
# print(nms_boxes.shape)
# a = np.array([]).reshape(-1,6)
# # print(a)
# # print(a.shape)
# b = 1
# a = ['1,2,3,4,5,6,7','1,2,3,4,5,6,7']
# print(len(a))
# # c = b//5
# print(c)
# a =
# path  = r"data-json"
# import os
# # for i in range(10):
# path1 = os.listdir(path)
# path1.sort()
# print(path1)
# # count = 1
# # a = [1,2,5,6,11,7,3,4,5,12,13]
# # print(a)
# # a.sort()
# # print(a)
# for i ,files in enumerate(path1):
# 	# if i < 9:
# 	# 	continue
# # 	# print(files)
# # 	file = open(os.path.join(path,files),encoding="utf-8").readlines()
# import os
# import PIL.Image as pimg
# label_path = r"label.txt"
# path = r"dataset_data"
# dataset = []
# dataset.extend(open(os.path.join(path,label_path)).readlines())
# for i  in range(14):
#
# 	img_path = os.path.join(path,dataset[i]).strip().split()
# 	img_data1  = pimg.open(img_path[0])
# 	print(img_data1)
# 	img_data1.show()a
# a = np.array([0,0,0,0,0,0])
# index = np.argmax(a)
# # print(index)
# a = np.arange(1,49).reshape(8,6)
# print(a)
# print(a[0:5])
# print(a[:,0:5])
#
# label_txt = open(r"test_image_txt/label.txt").readlines()
# print(label_txt)
# print(len(label_txt))
# list= [0,9,10,11,12,13,14,15,16,1,2,3,4,6,7]
# for  i  in range(15):
# 	print(i)
# 	label_txt1 = label_txt[i].strip().split()
# 	print(label_txt1)

a = np.arange(1,11)
b = np.split(a,indices_or_sections=len(a)//5)
print(b)