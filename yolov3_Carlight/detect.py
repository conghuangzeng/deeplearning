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
from  Nets import Main_net
# import cfg
import utils
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
#注意，只要长和宽是32的倍数就可以传入网络中使用
# net = torch.load(r"net1004_01_num1.pth")
# net = torch.load( r"net1006_mse_num15.pth")
net = torch.load( r"net1109.pth")
net = net.to(device)
net.eval()#必不可少的，理解
transform = transforms.Compose([
	transforms.Resize((416,416)),
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
		# print(output[...,0])

		index = iou>0.7#阈值设为0.5以上试试
		# print(index.shape)#(n,13,13,3)
		# print(output[iou>0.1])
		indexs = torch.nonzero(index)#(n,4的结构)#4指的是（N*13*13*3），理解。这里的n值得是大于0.5有多少个，这么理解。不是批次的N
		# print(indexs.shape)
		ves = output[index]#（n，15）的结构
		# print(ves.shape)
		return indexs,ves

	def parse(self,indexs,ves,anchors_group,side):
		anchors_group = torch.FloatTensor(anchors_group).to(device)
		anchors_rel_offset_iou_xy = ves[:,0:3]
		anchors_rel_offset_wh = ves[:, 3:5]
		# print(anchors_rel_offset.size())
		cls_es =ves[:,5:]
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
		cx = (cx_index.float()+anchors_rel_offset_iou_xy[:,1]+1.0)*side#索引+1才是当前格子数
		cy = (cy_index.float() + anchors_rel_offset_iou_xy[:, 2]+1.0)*side#比如索引为5是第六个格子
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
		cls_boxes =boxes[:, 5:]
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
			nms_boxes1 = utils.nms(boxes2,i = 0.3,isMin=False)#大
			# print(nms_boxes1)
			# nms_boxes1 = utils.nms(boxes2, i=0.3, isMin=False)#iou设为0.3两头鹿要丢一头，因为iou达到了0.55
			if nms_boxes1.shape[0]>0:
				lis_tatal.extend(nms_boxes1)
		# print(lis_tatal)
		nms_boxes = np.stack(lis_tatal)
		print(nms_boxes)
		return nms_boxes

	# print(a2.size())




	def main(self):
		test_path = r"test_img"
		for  j ,image_name in enumerate(os.listdir(test_path)):
			with pimg.open(os.path.join(test_path,image_name)) as  img:
				img = img.resize((416,416))
				img_data1  = transform(img)
				img_data1 = img_data1.to(device)
				img_data = torch.unsqueeze(img_data1,dim=0)
				# print(img_data.size())#(1,3,416,416)
				detetion_out_13, detetion_out_26, detetion_out_52 = net(img_data)
				# print(detetion_out_13.shape)#(1,45,13,13)
				indexs_13 ,ves_13 = self.filter(detetion_out_13)
				boxs_13 =self.parse(indexs_13,ves_13,anchor_groups[13],32)
				print(boxs_13.shape)
				indexs_26, ves_26 =self.filter(detetion_out_26)
				boxs_26 = self.parse(indexs_26, ves_26, anchor_groups[26],16)
				print(boxs_26.shape)
				indexs_52, ves_52 = self.filter(detetion_out_52)
				boxs_52 = self.parse(indexs_52, ves_52, anchor_groups[52],8)
				print(boxs_52.shape)
				boxs_total = torch.cat((boxs_13,boxs_26,boxs_52),dim=0)
				# print(boxs_total.shape)
				# print(boxs_total[:,0:5])
				nms_boxes_total = self.nms_box(boxs_total)
				print(nms_boxes_total.shape)#(N,6)的结构
				# print(nms_boxes_total.shape)
				img_draw  = Draw.ImageDraw(img)
				# 类别字典
				cls_dit = {0: "person", 1: "dog", 2: "horse", 3: "car", 4: "deer", 5: "ball", 6: "cat", 7: "elephant",8: "huaban", 9: "feipan"}  # 保存这个
				#字体定义
				font = Font.truetype("consola.ttf", size=10, encoding="utf-8")  # 设置字体

				#画实际框
				for box in nms_boxes_total:
					x1 = int(box[1])
					y1 = int(box[2])
					x2 = int(box[3])
					y2 = int(box[4])
					# print(x1,y1,x2,y2)
					img_draw.rectangle((x1,y1,x2,y2),outline="yellow",width=2)

					img_draw.rectangle((x1, y1-10, x2, y1), outline=r"#FF83FA", fill=r"#FF83FA",width=2)
					img_draw.text((x1,y1-10),text=cls_dit[int(box[5])],fill=r"#383838",font=font,anchor=(10,30))
					plt.scatter((x1+x2)/2,(y1+y2)/2,s=10,c="yellow",marker = "*")
				plt.imshow(img)
				plt.pause(2)
				plt.show()
				plt.cla()
				# img.save("test_result\{0}.jpg".format(label_txt[0][:3]))
				plt.close()
	def main1(self):
		test_path = r"test_img"
		for  j ,image_name in enumerate(os.listdir(test_path)):
			with pimg.open(os.path.join(test_path,image_name)) as  img:
				# img = img.resize((416,416))
				img_data1  = transform(img)
				img_data1 = img_data1.to(device)
				img_data = torch.unsqueeze(img_data1,dim=0)
				# print(img_data.size())#(1,3,416,416)
				detetion_out_13, detetion_out_26, detetion_out_52 = net(img_data)
				# print(detetion_out_13.shape)#(1,45,13,13)
				indexs_13 ,ves_13 = self.filter(detetion_out_13)
				boxs_13 =self.parse(indexs_13,ves_13,anchor_groups[13],32)
				print(boxs_13.shape)
				indexs_26, ves_26 =self.filter(detetion_out_26)
				boxs_26 = self.parse(indexs_26, ves_26, anchor_groups[26],16)
				print(boxs_26.shape)
				indexs_52, ves_52 = self.filter(detetion_out_52)
				boxs_52 = self.parse(indexs_52, ves_52, anchor_groups[52],8)
				print(boxs_52.shape)
				boxs_total = torch.cat((boxs_13,boxs_26,boxs_52),dim=0)
				# print(boxs_total.shape)
				# print(boxs_total[:,0:5])
				nms_boxes_total = self.nms_box(boxs_total)
				print(nms_boxes_total.shape)#(N,6)的结构
				# print(nms_boxes_total.shape)
				img_draw  = Draw.ImageDraw(img)
				# 类别字典
				cls_dit = {0: "person", 1: "dog", 2: "horse", 3: "car", 4: "deer", 5: "ball", 6: "cat", 7: "elephant",8: "huaban", 9: "feipan"}  # 保存这个
				#字体定义
				font = Font.truetype("consola.ttf", size=10, encoding="utf-8")  # 设置字体

				#画实际框
				for box in nms_boxes_total:
					x1 = int(box[1])
					y1 = int(box[2])
					x2 = int(box[3])
					y2 = int(box[4])
					# print(x1,y1,x2,y2)
					img_draw.rectangle((x1,y1,x2,y2),outline="yellow",width=2)

					img_draw.rectangle((x1, y1-10, x2, y1), outline=r"#FF83FA", fill=r"#FF83FA",width=2)
					img_draw.text((x1,y1-10),text=cls_dit[int(box[5])],fill=r"#383838",font=font,anchor=(10,30))
					plt.scatter((x1+x2)/2,(y1+y2)/2,s=10,c="yellow",marker = "*")
				plt.imshow(img)
				plt.pause(2)
				plt.show()
				plt.cla()
				# img.save("test_result\{0}.jpg".format(label_txt[0][:3]))
				plt.close()
if __name__ == '__main__':
	detect = Detector()
	detect.main()

