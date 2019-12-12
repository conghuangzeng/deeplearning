import torch
from torchvision import datasets, transforms
import torch.utils.data as data
import PIL.Image as pimg
from image_to_square import *
from detect_numpy import Detector
import os
import cv2
transform = transforms.Compose([
    # transforms.Resize((96,96)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# net = torch.load(r"resnet_1114.pth")
net = torch.load(r"resnet_star_115.pth")#识别明星
net = net.to(device)
net.eval()#加了BN层一定要有这个。
cls_dit = {0:"胡歌",1:"01",2:"憨豆",3:"03",4:"04",5:"05",6:"06",7:"07",8:"08",9:"09",10:"10",
          11:"11",12:"12",13:"13",14:"14",15:"15",16:"16",17:"17",18:"18",19:"19"}
img_path  = r"testimg_star"

# for files in os.listdir(img_path):
#     img  = pimg.open(os.path.join(img_path,files))
#     img = transform(img)
#     img = torch.unsqueeze(img,dim=0).to(device)
#     label = torch.tensor([1]).to(device)
#     output, cosa = net(img, label)
#     print(cosa*10)#cosa*10就是输入的图片与网络对比的相似度，因为arcloss求cosa的时候除以了10
#     #output是arcface激活后的输出，是做损失用的 。我们测试用的是激活前的cosa
#     index = torch.argmax(cosa,dim=1).item()
#     # print(index)
#     print(cls_dit[index])

#用opencv来做
for files in os.listdir(img_path):
    img  = cv2.imread(os.path.join(img_path,files))
    img = prep_image(img, inp_dim=96)

    #show出来
    # tf_toimg = transforms.ToPILImage()
    # img = tf_toimg(img)
    # img.show()
    img  = img.sub(0.5).div(0.5)
    img = torch.unsqueeze(img,dim=0).to(device)
    label = torch.tensor([1]).to(device)
    output, cosa = net(img, label)
    # print(cosa*10)#cosa*10就是输入的图片与网络对比的相似度，因为arcloss求cosa的时候除以了10
    #output是arcface激活后的输出，是做损失用的 。我们测试用的是激活前的cosa
    index = torch.argmax(cosa,dim=1).item()
    print(cosa[index])
    print(cls_dit[index])














