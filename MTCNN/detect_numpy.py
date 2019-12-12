import time
import torch
import  numpy as np
import nets
import PIL.Image as pimg
import PIL.ImageDraw as draw
import net_datas
from torchvision import  transforms
import utils
import os
import torch.utils.data as data
import matplotlib.pyplot as plt
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])


class Detector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #这三个网络路径效果将就，保存
        self.pnet  = torch.load(r"pnet0909.pth")
        self.rnet = torch.load(r"rnet0920.pth")
        self.onet = torch.load(r"onet0919.pth")#o网络0919留着，可以

        #
        # self.pnet  = torch.load(r"pnet0909.pth")
        # self.rnet = torch.load(r"rnet0924.pth")
        # self.onet = torch.load(r"onet0924.pth")
        self.pnet = self.pnet.to(self.device)
        self.rnet = self.rnet.to(self.device)
        self.onet = self.onet.to(self.device)


    def detect(self,image):
        self.pnet.eval()#对应网络层的batchnormal
        self.rnet.eval()
        self.onet.eval()
        p_start_time = time.time()
        p_boxes = self.detect_pnet(image)
        if p_boxes.shape[0]<1:
            return np.array([])
        p_end_time = time.time()

        pnet_time = p_end_time - p_start_time
        # return p_boxes
        # return None
        r_start_time = time.time()
        r_boxes = self.detect_rnet(image,p_boxes)
        if r_boxes.shape[0] < 1:
            return np.array([])

        r_end_time = time.time()

        rnet_time = r_end_time - r_start_time
        # return None
        # return r_boxes
        # # # #
        o_start_time = time.time()
        o_boxes = self.detect_onet(image,r_boxes)
        if o_boxes.shape[0] < 1:
            return np.array([])
        o_end_time = time.time()

        onet_time = o_end_time - o_start_time

        total_time  = pnet_time + rnet_time + onet_time

        print("total_time:{0} = pnet_time:{1}+ rnet_time:{2}+onet_time:{3}".format(total_time, pnet_time, rnet_time,onet_time))
        # return None
        return o_boxes




    def detect_pnet(self,image):
        scale = 1
        w ,h = image.size
        _w,_h = w,h
        min_side_len = min(_w,_h)
        boxes = []
        img = image

        while min_side_len>12:
            img_data = transform(img)
            img_data.unsqueeze_(0)#转换成1,c,h,w
            img_data = img_data.to(self.device)
            cond, offset = self.pnet(img_data)
            offset=offset.detach()#变量能求导，提取变量元素，变为标量,结构(N4HW)
            cond = cond.detach()#结构(N1HW)
            _cond,_offset = cond[0][0].cpu(),offset[0].cpu()
            indexs  = torch.nonzero(torch.gt(_cond,0.6))#(N,2)
            offset_boxes = self.offset_to_boxes(indexs,_cond,_offset,scale)
            scale *= 0.7#缩放比例这个很关键，搞成0.9有些脸要漏掉
            _w,_h = int(w*scale),int(h*scale)
            min_side_len = min(_w,_h)
            img= img.resize((_w, _h))
            _boxes= utils.nms(offset_boxes,i=0.5,isMin=False)
            boxes.extend(_boxes)
        p_boxes = np.array(boxes)

        return p_boxes

    def detect_rnet(self,image,p_boxes):
        _p_boxes = self.convert_to_square(p_boxes)
        a_boxes = []
        count = 1
        for i,r_box in enumerate(_p_boxes):#简化思路：裁剪图片用矩阵的思想，难度大，图片本身就是一个矩阵，这样速度会更快
            img = image.crop((int(r_box[0]),int(r_box[1]),int(r_box[2]),int(r_box[3])))
            # img.save("r_test_image\{0}.jpg".format(r_box[4]))
            img = img.resize((24,24))
            # plt.imshow(img)
            # plt.pause(1)
            # img.save(r"r_test_image\rtest{0}.jpg".format(str(count)))
            img_data_r = transform(img)
            a_boxes.append(img_data_r)
            count+=1
        img_dataset = torch.stack(a_boxes)
        # print(_p_boxes.shape)#p的框数量

        img_dataset = img_dataset.to(self.device)
        cond, offset = self.rnet(img_dataset)
        offset = offset.detach()
        cond = cond.detach()
        cond, offset = cond.cpu().numpy(), offset.cpu().numpy()#(n,1)(n,4)
        # print(cond)
        indexs,_ = np.where(cond>0.6)#(m,2)
        #建议框的位置
        x11 = np.int16(_p_boxes[indexs,0])
        y11 = np.int16(_p_boxes[indexs,1])
        x22 = np.int16(_p_boxes[indexs,2])
        y22 = np.int16(_p_boxes[indexs,3])
        max_side_len =np.maximum(x22 - x11, y22 - y11)
        #真实框的位置
        x1 = np.int16(offset[indexs,0]*max_side_len+x11)
        y1 =np.int16(offset[indexs,1]*max_side_len+y11)
        x2 = np.int16(offset[indexs, 2]*max_side_len+x22)
        y2= np.int16(offset[indexs,3]*max_side_len+y22)
        # print(cond.shape)
        _cond = cond[indexs][:,0]
        # print(_cond.shape)

        c_boxes = np.array([x1,y1,x2,y2,_cond]).T

        # print(c_boxes.shape)#(n,5)
        # print(c_boxes)#R的框数量

        if c_boxes.shape[0]==0:
            return np.array([])
        # path = image_path
        # test_img_r= pimg.open(path)
        #
        # for c_box in c_boxes:

            # test_img_r1 = test_img_r.crop((c_box[0],c_box[1],c_box[2],c_box[3]))
            # test_img_r1.save("r_test_image\{0}.jpg".format(c_box[4]))
        # r_boxes  =  c_boxes
        r_boxes = utils.nms(np.array(c_boxes),0.5,isMin=False)
        return r_boxes


    def detect_onet(self,image,r_boxes):
        _r_boxes = self.convert_to_square(r_boxes)
        # print(_r_boxes.shape)
        a_boxes = []
        for o_box in _r_boxes:

            img = image.crop((int(o_box[0]),int(o_box[1]),int(o_box[2]),int(o_box[3])))
            img = img.resize((48,48))
            # img_r_draw = draw.ImageDraw(image)
            # img_r_draw.rectangle((int(o_box[0]),int(o_box[1]),int(o_box[2]),int(o_box[3])),outline="green",width=4)

            img_data_o = transform(img)
            a_boxes.append(img_data_o)
        img_dataset = torch.stack(a_boxes)
        img_dataset = img_dataset.to(self.device)

        cond, offset = self.onet(img_dataset)
        offset = offset.detach()  #
        cond = cond.detach()
        cond, offset = cond.cpu().numpy(), offset.cpu().numpy()
        indexs,_= np.where(cond>0.9)
        # 建议框的位置
        x11 = np.int16(_r_boxes[indexs, 0])
        y11 = np.int16(_r_boxes[indexs, 1])
        x22 = np.int16(_r_boxes[indexs, 2])
        y22 = np.int16(_r_boxes[indexs, 3])
        max_side_len = np.maximum(x22 - x11, y22 - y11)
        # 真实框的位置
        x1 = np.int16(offset[indexs, 0] * max_side_len + x11)
        y1 = np.int16(offset[indexs, 1] * max_side_len + y11)
        x2 = np.int16(offset[indexs, 2] * max_side_len + x22)
        y2 = np.int16(offset[indexs, 3] * max_side_len + y22)
        _cond = cond[indexs][:, 0]

        c_boxes = np.array([x1, y1, x2, y2, _cond]).T
        if c_boxes.shape[0] == 0:
            return np.array([])


        o_boxes = utils.nms(np.array(c_boxes),0.3,isMin=True)
        return o_boxes


    def convert_to_square(self,boxes):
        square_boxes = boxes.copy()#首先先复制boxes的形状，这种编辑函数的思路理解一下。
        if boxes.shape[0]<1:
            return np.array([])
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        cond = boxes[:,4]
        cx = (x1+x2)/2
        cy = (y1 + y2) / 2
        w,h = x2-x1,y2-y1
        max_side_len = np.maximum(w,h)
        square_boxes[:,0] = cx - max_side_len/2
        square_boxes[:,1]= cy-max_side_len/2
        square_boxes[:,2] = square_boxes[:,0]+max_side_len
        square_boxes[:,3] =  square_boxes[:,1]+max_side_len
        square_boxes[:, 4] = cond
        return square_boxes



    def offset_to_boxes(self,index,_cond,_offset,scale,side_len = 12,stride=2):
        #建议框的位置
        index=index.numpy()
        _cond=_cond.numpy()
        _offset=_offset.numpy()#4*m*n的结构
        x11 = np.int16(((index[:,1]*stride))/scale)#必须加float，整型除以scale浮点型会有报警
        y11 = np.int16(index[:,0]*stride/scale)
        x22= np.int16((index[:,1]*stride+side_len)/scale)
        y22 = np.int16((index[:,0]*stride+side_len)/scale)
        ow = x22-x11
        oh = y22-y11
        #真实框的位置
        __offset = _offset[:,index[:,0],index[:,1]].T
        # print(__offset.shape)#(N,4)
        x1 = __offset[:,0]* ow+x11
        # print(x1.shape)
        y1 = __offset[:,1] * oh + y11
        x2 = __offset[:,2] *  ow + x22
        y2 = __offset[:,3] *  oh + y22
        boxes  = np.array((x1,y1,x2,y2,_cond[index[:,0],index[:,1]])).T
        # print(type(box))
        # print(box.shape)
        return boxes



if __name__ == '__main__':
    path = r"D:\AIpractise\Projects\20190822_MTCNN\test_image"
    detector = Detector()

    for image in os.listdir(path):
            # print(os.listdir(path))
            # print(image)
        image_path = os.path.join(path, image)
        print(image_path)
        img1 = pimg.open(image_path)
        img1 = img1.convert("RGB")
        boxes = detector.detect(img1)
        draw_img1 = draw.ImageDraw(img1)
        count = str(image_path[-7:-4])
        print(count)
        for box in boxes:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            # print(box[4])
            draw_img1.rectangle((x1,y1,x2,y2),outline="red",width=3)
            # img1.save(r"test_result\test01{}.jpg".format(count))
        plt.imshow(img1)
        plt.show()
        plt.pause(1)


