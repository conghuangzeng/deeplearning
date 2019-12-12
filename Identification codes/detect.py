import os
import numpy as np
import torch
import torch.utils.data as data
from torchvision import  transforms
from  nets import SEQ2SEQ
from nets_data import Sampling
import torch.nn as nn
import PIL.Image as pimg
import math
import matplotlib.pyplot as plt
import PIL.ImageDraw as Draw
import PIL.ImageFont as Font

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])
net = torch.load("net.pth").to(device)
font = Font.truetype("consola.ttf", size=30, encoding="utf-8")
test_path = r"test_img"
acc_total = 0
for  j ,image_name in enumerate(os.listdir(test_path)):
    with pimg.open(os.path.join(test_path,image_name)) as  img:
        print(image_name)
        image_name = image_name.split(".")
        label = np.array(image_name[1:5],dtype=np.int32)
        print(label)
        img_data1  = transform(img)
        img_data1 = img_data1.to(device)
        img_data = torch.unsqueeze(img_data1,dim=0)
        output = net(img_data)
        # print(output.shape)
        output1 = torch.argmax(output,dim=2).squeeze(0).cpu().detach().numpy()
        print(output1)
        img_draw = Draw.ImageDraw(img)
        text = "{0} {1} {2} {3} ".format(output1[0],output1[1],output1[2],output1[3])
        img_draw.text((0, 0), text=text, fill=(255,255,255), font=font)
        # plt.text(x=10,y=10,s=30,)
        # plt.imshow(img)
        # plt.pause(2)
        # plt.show()
        # plt.cla()
        acc = np.sum( output1 == label, dtype=np.float32) / (1 * 4)
        acc_total+=acc
print("acctotal:{:.3f}%".format( acc_total * len(os.listdir(test_path))))
        # label = label.float()
            # print(img_data.size())#(1,3,416,416)