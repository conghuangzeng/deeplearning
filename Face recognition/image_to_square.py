import PIL.Image as pimg
import cv2
import numpy as np
import torch
from torchvision import transforms

import os
count = 1

def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
	#按照最长边来进行缩放，416/最长边得到的缩放比例最小，理解。
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
	#创建一个416*416的灰度图，再把调整后的图片沾上去，牛逼啊 。！
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image

    return canvas

def prep_image(img, inp_dim):#
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """
    #这里是传唤为CHW的tensor，在用toPILIMAge进行还原
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:,:,::-1].transpose((2,0,1)).copy()##将BGR换成RGB，换通道
    # img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)


    # img= (letterbox_image(img, (inp_dim, inp_dim)))
    # img= img[:,:,::-1]
    img = torch.from_numpy(img).float().div(255.0)
    return img

if __name__ == '__main__':
    path = r"D:\shujuji\star face net\huojianhua0"
    #opencv不支持中文的文件名，我擦 。注意这点
    tf_toimg = transforms.ToPILImage()
    inp_dim = 96
    count= 1
    for images in  os.listdir(path):
        print(os.path.join(path,images))
        image  = cv2.imread(os.path.join(path,images))
        # print(image.shape)
        # cv2.imshow("sss",image)
    # image = cv2.imread(r"6.jpg")
        img = prep_image(image,inp_dim)
        img  = tf_toimg(img)
        # img.show()
        img.save(r"D:\shujuji\star face net\train_img\1huojianhua\{0}.jpg".format(count))
        count+=1


