import time
import torch
import  numpy as np
import nets
import PIL.Image as pimg
import PIL.ImageDraw as draw
import matplotlib.pyplot as plt
import net_datas
from torchvision import  transforms
import utils
import torch.utils.data as data
transform = transforms.Compose([
    transforms.ToTensor()
])


class Detector:
    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pnet  = torch.load(r"D:\AIpractise\Projects\20190822_MTCNN\pnet.pth")
        self.rnet = torch.load(r"D:\AIpractise\Projects\20190822_MTCNN\rnet.pth")
        self.onet = torch.load(r"D:\AIpractise\Projects\20190822_MTCNN\onet.pth")
        self.pnet = self.pnet.to(self.device)
        self.rnet = self.rnet.to(self.device)
        self.onet = self.onet.to(self.device)




    def detect(self,image):
        p_start_time = time.time()
        p_boxes = self.detect_pnet(image)
        # print(self.p_boxes)
        if p_boxes.shape[0]<1:
            return np.array([])
        p_end_time = time.time()

        pnet_time = p_end_time - p_start_time

        r_start_time = time.time()
        r_boxes = self.detect_rnet(image,p_boxes)
        if r_boxes.shape[0] < 1:
            return np.array([])
        r_end_time = time.time()

        rnet_time = r_end_time - r_start_time
        # return r_boxes
        # return None
        o_start_time = time.time()
        o_boxes = self.detect_onet(image,r_boxes)
        if o_boxes.shape[0] < 1:
            return np.array([])
        o_end_time = time.time()

        onet_time = o_end_time - o_start_time

        total_time  = pnet_time + rnet_time + onet_time

        print("total_time:{0} = pnet_time:{1}+ rnet_time:{2}+onet_time:{3}".format(total_time, pnet_time, rnet_time,onet_time))

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
            img_data.unsqueeze_(0)
            img_data = img_data.to(self.device)
            # print(img_data.size())
            cond, offset = self.pnet(img_data)
            offset=offset.detach()#
            cond = cond.detach()
            _cond,_offset = cond[0][0].cpu(),offset[0].cpu()
            indexs  = torch.nonzero(torch.gt(_cond,0.6))
            for index in indexs:
                boxes.append(self.offset_to_boxes(index,_cond[index[0],index[1]],_offset,scale))
            scale *= 0.7
            _w,_h = int(w*scale),int(h*scale)
            min_side_len = min(_w,_h)
            img= img.resize((_w, _h))
        if len(boxes)==0:
            return np.array([])
        p_boxes= utils.nms(np.array(boxes),i=0.5,isMin=False)


        return p_boxes

    def detect_rnet(self,image,p_boxes):
        _p_boxes = self.convert_to_square(p_boxes)
        # print(_p_boxes.shape)
        a_boxes = []
        for r_box in _p_boxes:

            img = image.crop((int(r_box[0]),int(r_box[1]),int(r_box[2]),int(r_box[3])))
            img = img.resize((24,24))
            # plt.imshow(img_r)
            img_data_r = transform(img)

            a_boxes.append(img_data_r)
        img_data_r = torch.stack(a_boxes)
        img_data_r = img_data_r.to(self.device)
        cond, offset = self.rnet(img_data_r)
        offset = offset.detach()
        cond = cond.detach()
        _cond, _offset = cond.cpu().numpy(), offset.cpu().numpy()
        indexs , _ = np.where(_cond>0.6)

        c_boxes = []
        # print(indexs.size())
        for index in indexs:
            # print(index)
        #     #建议框的位置
            x11 = int(_p_boxes[index][0])
            y11 = int(_p_boxes[index][1])
            x22 = int(_p_boxes[index][2])
            y22 = int(_p_boxes[index][3])

            max_side_len = max(x22-x11,y22-y11)
            #原框的位置

            x1 =int(offset[index][0]*max_side_len+x11)
            y1 =int(offset[index][1] * max_side_len + y11)
            x2 =int(offset[index][2]*max_side_len+x22)
            y2 =int(offset[index][3]*max_side_len+y22)
            _cond = cond[index][0]
            c_boxes.append([x1,y1,x2,y2,_cond])
        if len(c_boxes)==0:
            return np.array([])
        r_boxes = utils.nms(np.array(c_boxes),0.5,isMin=False)
        return r_boxes


    def detect_onet(self,image,r_boxes):
        _r_boxes = self.convert_to_square(r_boxes)
        # print(_r_boxes.shape)
        a_boxes = []
        for o_box in _r_boxes:

            img = image.crop((int(o_box[0]),int(o_box[1]),int(o_box[2]),int(o_box[3])))
            img = img.resize((48,48))
            # plt.imshow(img_r)
            img_data_o = transform(img)

            a_boxes.append(img_data_o)
        img_data_o = torch.stack(a_boxes)
        img_data_o = img_data_o.to(self.device)
        # print(img_data_o.size())
        cond, offset = self.onet(img_data_o)
        # return _p_boxes.shape

        # print(cond.size())
        # print(offset.size())
        offset = offset.detach()  #
        cond = cond.detach()
        _cond, _offset = cond.cpu().numpy(), offset.cpu().numpy()
        indexs,_= np.where(_cond>0.5)

        c_boxes = []
        # print(indexs.size())
        for index in indexs:
            # print(index)
        #     #建议框的位置
            x11 = int(_r_boxes[index][0])
            y11 = int(_r_boxes[index][1])
            x22 = int(_r_boxes[index][2])
            y22 = int(_r_boxes[index][3])

            max_side_len = max(x22-x11,y22-y11)
            #原框的位置

            x1 =int(offset[index][0]*max_side_len+x11)
            y1 =int(offset[index][1] * max_side_len + y11)
            x2 =int(offset[index][2]*max_side_len+x22)
            y2 =int(offset[index][3]*max_side_len+y22)
            _cond = cond[index][0]
            c_boxes.append([x1,y1,x2,y2,_cond])
        print(c_boxes)
        if len(c_boxes)==0:
            return np.array([])
        o_boxes = utils.nms(np.array(c_boxes),0.3,isMin=True)
        return o_boxes


    def convert_to_square(self,boxes):
        square_boxes = boxes.copy()
        if boxes.shape[0]<1:
            return np.array([])
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        # print(x1,y1)
        cond = boxes[:,4]
        cx = (x1+x2)/2
        cy = (y1 + y2) / 2
        w,h = x2-x1,y2-y1
        max_side_len = np.maximum(w,h)
        # _x1 = cx-max_side_len
        # _y1 = cy-max_side_len
        # _x2 = _x1+max_side_len
        # _y2 = _y1+max_side_len
        square_boxes[:,0] = cx - max_side_len
        square_boxes[:,1]= cy-max_side_len
        square_boxes[:,2] = square_boxes[:,0]+max_side_len
        square_boxes[:,3] =  square_boxes[:,1]+max_side_len
        square_boxes[:, 4] = cond
        return square_boxes



    def offset_to_boxes(self,index,_cond,_offset,scale,side_len = 12,stride=2):
        #建议框的位置

        x11 = int(float(index[1].item()*stride)/scale)#必须加float，整型除以scale浮点型会有报警
        y11 = int(float(index[0].item() * stride) / scale)
        x22= int(float(index[1].item()*stride+side_len)/scale)
        y22 = int(float(index[0].item()*stride+side_len)/scale)
        ow = x22-x11
        oh = y22-y11
        # print(w,h)
        # print(12/scale)
        # max_side_len = max(ow,oh)

        #真实框的位置
        __offset = _offset[:,index[0],index[1]]#4*m*n的结构
        x1 = __offset[0].item()* ow+x11
        y1 = __offset[1].item() * oh + y11
        x2 = __offset[2].item() *  ow + x22
        y2 = __offset[3].item() *  oh + y22
        # print([x1,y1,x2,y2,_cond.item()])
        return np.array([x1,y1,x2,y2,_cond])



if __name__ == '__main__':
    image_path = r"D:\AIpractise\Projects\20190822_MTCNN\test_image\test02.jpg"
    detector = Detector()
    with pimg.open(image_path) as img1:
        boxes = detector.detect(img1)
        draw_img1 = draw.ImageDraw(img1)
        # print(boxes.shape)
        for box in boxes:

            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            # print(box[4])
            draw_img1.rectangle((x1,y1,x2,y2),outline="red",width=1)
        img1.show()


