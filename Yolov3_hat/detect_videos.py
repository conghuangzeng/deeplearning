import torch
# import torch.utils.data as data
import os
import torch.nn as nn
from  torchvision import transforms
import PIL.Image as pimg
import numpy as np
import math
from Nets import Main_net
from  Nets_data import Yolodataset
import matplotlib.pyplot as plt
import PIL.ImageDraw as Draw
import PIL.ImageFont as Font
import cv2
from img_to_square import prep_image,letterbox_image
from  Nets import Main_net
# import cfg
import utils
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
#注意，只要长和宽是32的倍数就可以传入网络中使用
net = torch.load( r"net1106.pth")
net = net.to(device)
net.eval()#必不可少的，理解
transform = transforms.Compose([
	# transforms.Resize((416,416)),
	transforms.ToTensor(),
	transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]
	)
sigmoid = nn.Sigmoid()
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
anchors = anchor_groups

class Detector:

	def filter(self,output):
		output = output.permute(0,2,3,1)
		output = output.reshape(output.size()[0],output.size()[1],output.size()[2],3,-1)
		iou = output[...,0]
		index = iou>0.6#阈值设为0.5以上试试
		indexs = torch.nonzero(index)#(n,4的结构)#4指的是（N*13*13*3），理解。这里的n值得是大于0.5有多少个，这么理解。不是批次的N
		ves = output[index]#（n，15）的结构
		return indexs,ves

	def parse(self,indexs,ves,anchors_group,side,new_w,new_h,w):
		anchors_group = torch.FloatTensor(anchors_group).to(device)
		anchors_rel_offset_iou_xy = ves[:,0:3]
		anchors_rel_offset_wh = ves[:, 3:5]
		# print(anchors_rel_offset.size())
		cls_es =ves[:,5:15]
		# print(cls_es.size())
		#图片
		img_num = indexs[:,0]#这里传入的是一张图片，这里没用。
		cx_index = indexs[:,2]
		cy_index = indexs[:,1]
		#建议框的3种形状
		anchors_index= indexs[:,3]#（N的结构）
		# print(anchors_index.shape)
		# 	# print(anchors_index)
		iou = anchors_rel_offset_iou_xy[:,0]
		cx = (cx_index.float()+anchors_rel_offset_iou_xy[:,1]+1.0)*side-(416-new_w)*0.5#索引+1才是当前格子数
		cy = (cy_index.float() + anchors_rel_offset_iou_xy[:, 2]+1.0)*side-(416-new_h)*0.5
		#比如索引为5是第六个格子
		cx = cx*w/new_w
		cy = cy * w / new_w
		w = torch.exp(anchors_rel_offset_wh[:,0])*anchors_group[anchors_index,0]#这个不会报错，因为anchors_index也是根据indexs来的
		h =  torch.exp(anchors_rel_offset_wh[:,1])*anchors_group[anchors_index,1]
		boxes1 =torch.stack(([iou,cx,cy,w,h]),dim=0).permute(1,0)#（N,5）
		# print(boxes1.shape)
		boxes = torch.cat((boxes1,cls_es),dim=1)
		# print(boxes.shape)#(N,15)
		return boxes

	def nms_box(self,boxes):
		if boxes.shape[0]==0:
			return np.array([]).reshape(-1,6)
		lis_tatal = []
		cls_boxes =boxes[:, 5:15]
		index = torch.argmax(cls_boxes, dim=1).float()
		index = index.reshape(-1, 1)
		_boxes= torch.cat((boxes[:,0:5],index), dim=1)#(n,6)
		_boxes = _boxes.cpu().detach()
		_boxes = _boxes.numpy()
		# print(_boxes.shape)
		# print(_boxes)
		for i in range(10):
			index = np.where(_boxes[:, 5] == i)
			boxes1 = _boxes[index]
			boxes2 = boxes1.copy()
			boxes2[:,0] = boxes1[:,0]
			boxes2[:, 1] = boxes1[:,1]-boxes1[:,3]*0.5
			boxes2[:,2] = boxes1[:,2]-boxes1[:,4]*0.5
			boxes2[:,3] = boxes2[:, 1]+boxes1[:,3]
			boxes2[:, 4] =  boxes2[:, 2]+boxes1[:,4]
			boxes2[:, 5] = boxes1[:,5]
			# print(boxes2)
			nms_boxes1 = utils.nms(boxes2, i = 0.3, isMin=False)#大
			# print(nms_boxes1)
			# nms_boxes1 = utils.nms(boxes2, i=0.3, isMin=False)#iou设为0.3两头鹿要丢一头，因为iou达到了0.55
			if nms_boxes1.shape[0]>0:
				lis_tatal.extend(nms_boxes1)
		# print(lis_tatal)
		nms_boxes = np.stack(lis_tatal)
		# print(nms_boxes.shape)
		return nms_boxes

	# print(a2.size())




	def main(self,img):
		img_pil = img[:,:,::-1].copy()
		img_pil = pimg.fromarray(img_pil)#转换为图片格式
		# print(type(img_pil))
		h, w ,_=img.shape
		img_416, new_w, new_h = letterbox_image(img, (416, 416))
		img_416 = img_416[:, :, ::-1].transpose((2,0,1)).copy()
		##将BGR换成RGB，换通道
		img_416 = torch.from_numpy(img_416).float().div(255.0).sub(0.5).div(0.5)
		img_data1 = img_416.to(device)
		img_data = torch.unsqueeze(img_data1,dim=0)
		# print(img_data.size())#(1,3,416,416)
		detetion_out_13, detetion_out_26, detetion_out_52 = net(img_data)
		indexs_13 ,ves_13 = self.filter(detetion_out_13)
		boxs_13 =self.parse(indexs_13,ves_13,anchor_groups[13],32,new_w,new_h,w)
		indexs_26, ves_26 =self.filter(detetion_out_26)
		boxs_26 = self.parse(indexs_26, ves_26, anchor_groups[26],16,new_w,new_h,w)
		indexs_52, ves_52 = self.filter(detetion_out_52)
		boxs_52 = self.parse(indexs_52, ves_52, anchor_groups[52],8,new_w,new_h,w)
		boxs_total = torch.cat((boxs_13,boxs_26,boxs_52),dim=0)
		nms_boxes_total = self.nms_box(boxs_total)
		# print(nms_boxes_total.shape)#(N,6)的结构
		return nms_boxes_total

if __name__ == '__main__':
	detect = Detector()
	detect.main()

