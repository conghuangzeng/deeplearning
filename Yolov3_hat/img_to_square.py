import PIL.Image as pimg
import cv2
import numpy as np
import torch
from torchvision import transforms
import os
count = 1
path = r"C:\Users\Admin\Desktop\imgdata\image（初减）"
def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
	#按照最长边来进行缩放，416/最长边得到的缩放比例最小，理解。
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
	#创建一个416*416的灰度图，再把调整后的图片沾上去，牛逼！
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image

    return canvas,new_w,new_h

def prep_image(img, inp_dim):#
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """
    img,new_w,new_h = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:,:,::-1].transpose((2,0,1)).copy()##将BGR换成RGB，换通道
    # img = torch.from_numpy(img).float().div(255.0)
    img = torch.from_numpy(img).float()
    # img = img[:, :, ::-1].copy()
    return img

if __name__ == '__main__':
    tf_toimg = transforms.ToPILImage()
    image  = cv2.imread(r"dataset/10.jpg")
    inp_dim = 416
    # print(inp_dim)
    img,new_w,new_h = letterbox_image(image,(416,416))#是个数组
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()  ##将BGR换成RGB，换通道
    img = torch.from_numpy(img).float().div(255.0)
    print(type(img))
    img  = tf_toimg(img)
    print(type(img))
    img.show()
