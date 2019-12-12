import torch
from torchvision import datasets, transforms
import torch.utils.data as data
from image_to_square import *
from detect_numpy import Detector
import PIL.Image as pimg
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
net.eval()#加了BN层一定要有这个。
cls_dit = {0:"huge",1:"茅小春",2:"女二",3:"03",4:"04",5:"05",6:"06",7:"07",8:"08",9:"09",10:"10",
          11:"11",12:"12",13:"13",14:"14",15:"15",16:"16",17:"17",18:"18",19:"19"}

path = r"C:\Users\admin\Desktop\huge1.mp4"
cap = cv2.VideoCapture(path)#打开视频文件
detector  = Detector()
fps = cap.get(cv2.CAP_PROP_FPS)#FPS帧数，帧数指每一秒有多少张图片
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 保存视频的编码
out = cv2.VideoWriter('7.avi', fourcc, 20.0, (int(cap.get(3)),int(cap.get(4))))
count = 0
i=0
boxes=""
while True:
    ret,fram = cap.read()
    if ret:#ret是bool值，理解
        _image = cv2.cvtColor(fram, cv2.COLOR_BGR2RGB)#必须要转换，不然传到网络里面去精度太低了。
        image = fram
        if count%4==0 and count>0:
            boxes = detector.detect(_image)
            # print(boxes.shape)
        for box in boxes:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0))
            img = _image[y1:y2,x1:x2]
            img = prep_image(img,inp_dim = 96)
            # print(img.shape)
            img = img.sub(0.5).div(0.5)
            img = torch.unsqueeze(img, dim=0).to(device)
            # label = torch.tensor([1]).to(device)
            label = torch.arange(1).cuda()
            output,cosa = net(img,label)
            index= torch.argmax(cosa,dim=1).item()
            # print(cosa.shape)
            #
            # print(index)
            text = str((cosa[0,index]).float().item()*10)[:5]
            text1 = str(cls_dit[index])
            cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 255, 255), 2)
            cv2.putText(image, text1, (x1,y1-30), cv2.FONT_HERSHEY_PLAIN, 3.0, (0,0,0),2)


        cv2.imshow("show",image)#原图上画框，show的也是原图
        # cv2.waitKey(1)#等待时间，毫秒
        # out.write(fram)#存储每一帧的内容
        count += 1
        i+=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()#释放cap视频文件
cv2.destroyAllWindows()#清除所有窗口的东西














