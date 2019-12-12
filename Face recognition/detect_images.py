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
net = torch.load(r"resnet_star_1115.pth")#识别明星
net = net.to(device)
detector  = Detector()
net.eval()#加了BN层一定要有这个。
cls_dit = {0:"huge",1:"huojianhua",2:"憨豆",3:"03",4:"04",5:"05",6:"06",7:"07",8:"08",9:"09",10:"10",
          11:"11",12:"12",13:"13",14:"14",15:"15",16:"16",17:"17",18:"18",19:"19"}
img_path  = r"testimg_star"


#用opencv来做,图片尺寸不受限制
for files in os.listdir(img_path):
    image  = cv2.imread(os.path.join(img_path,files))

    _image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = detector.detect(_image)
    if len(boxes) ==0:
        continue
    #show出来
    # tf_toimg = transforms.ToPILImage()
    # img = tf_toimg(img)
    # img.show()

    for box in boxes:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0))
        img = image[y1:y2, x1:x2]
        # print(img.shape)
        if img.shape[0]==0 or  img.shape[1]==0:
            continue
        img = prep_image(img, inp_dim=96)
        img = img.sub(0.5).div(0.5)
        img = torch.unsqueeze(img, dim=0).to(device)
        label = torch.arange(1).cuda()
        output, cosa = net(img, label)
        index = torch.argmax(cosa, dim=1).item()

        # print(cosa*10)#cosa*10就是输入的图片与网络对比的相似度，因为arcloss求cosa的时候除以了10
        #output是arcface激活后的输出，是做损失用的 。我们测试用的是激活前的cosa

        print((cosa[0,index]).float().item()*10)
        print(cls_dit[index])
        text = str((cosa[0, index]).float().item() * 10)[:5]
        text1 = str(cls_dit[index])
        cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 255, 255), 2)
        cv2.putText(image, text1, (x1, y1+100), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 0), 2)

    cv2.imshow("show", image)  # 原图上画框，show的也是原图
    cv2.waitKey(0)#等待时间，毫秒
    cv2.destroyAllWindows()  # 清除所有窗口的东西














