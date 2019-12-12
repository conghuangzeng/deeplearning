import torch
import torch.utils.data as data
import os
from  torchvision import transforms
# import cfg
import PIL.Image as pimg
import numpy as np
import math
#比较难，重点理解
transform = transforms.Compose([
	transforms.Resize((416,416)),
	transforms.ToTensor(),
	transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
	]
	)

img_h = 416
img_w = 416

cls_num = 1
#搞成字典的额形式更好用
anchor_groups = {
	13:[[116,90],[156,198],[373,326]],
	26:[[30,61],[62,45],[59,119]],
	52:[[10,13],[16,30],[33,23]]}
#先验框尺寸(10x13)，(16x30)，(33x23)，(30x61)，(62x45)，(59x119)，(116x90)，(156x198)，(373x326)。
anchor_area_groups = {
	13:[x*y for x,y in anchor_groups[13]],
	26:[x*y for x,y in anchor_groups[26]],
	52:[x*y for x,y in anchor_groups[52]]}

cls_dit = {0: "person", 1: "dog", 2: "horse", 3: "car", 4: "deer", 5: "ball", 6: "cat", 7: "elephant",
				   8: "huaban", 9: "feipan"}#保存这个
#归一化函数

label_path = r"label.txt"

def one_hot(cls_num,s):
	result = np.zeros(shape=(cls_num))
	result[s]=1.
	return result
class Yolodataset(data.Dataset):
	def __init__(self,path,transform):
		self.path = path
		self.transform = transform
		self.dataset = []
		self.dataset.extend(open(os.path.join(path,label_path)).readlines())
	def __len__(self):
		return  len(self.dataset)
	def __getitem__(self, index):
		img_path = os.path.join(self.path,self.dataset[index]).strip().split()
		img_data1  = pimg.open(img_path[0])
		# print(img_path[0])
		img_data  = transform(img_data1)
		_boxes = np.array([float(x) for  x in img_path[1:]])
		boxes =  np.split(_boxes,indices_or_sections=len(_boxes)//5)
		# print(boxes)
		label={}#注意label是个列表
		# feature_sizes = [13,26,52]
		cls_num = 10
		for feature_size,anchors in anchor_groups.items():
			label[feature_size]=np.zeros(shape=(feature_size,feature_size,3,5+cls_num))
			for box in boxes:
				cls,cx,cy,b_w,b_h = box#cls是类别，是一个数
				offset_cx,index_cx = math.modf(cx*feature_size/416)
				offset_cy, index_cy = math.modf(cy*feature_size/416)
				for i ,anchor in enumerate(anchors):
					anchor_area = anchor_area_groups[feature_size][i]
					offset_w = np.log(b_w/anchor[0])
					offset_h = np.log(b_h / anchor[1])
					area = b_w*b_h
					# print(b_w,b_h)
					iou = min(area,anchor_area)/max(area,anchor_area)#这句有问题
					# print(int(index_cy-1),int(index_cx-1))
					label[feature_size][int(index_cy-1),int(index_cx-1),i] =[iou,offset_cx,offset_cy,offset_w,offset_h,*one_hot(cls_num,int(cls))]#根据造标签的特点来看，特征图上中心点所在的格子iou才是大于0 的，其他的格子iou全是0

		return img_data,label[13],label[26],label[52]
		# return img_data
#根据造标签的特点来看，特征图上中心
if __name__ == '__main__':
	##每次写完dataset确保要跑通
	yolodataset = Yolodataset(r"dataset_data",transform=transform)
	print(yolodataset)
	# img =
	tf_toimage = transforms.ToPILImage()
	for i in range(14):
		if i >0:
			break
		img1_data = (yolodataset[i])#1对应__getitem__(self, index)的index,就是第几张图
		img_data = img1_data[0]
		print(img_data.shape)
		print("13")
		iou = img1_data[1]
		print(iou.shape)
		# index =iou>0
		# # print(index)
		# print(iou[index])



