import torch
import  numpy as np


# a = torch.arange(24).reshape(6,4)
# print(a)
# b = torch.arange(9).reshape(9,1)
# print(b)
# # indexs = torch.lt(b,5)
# # file_path='c:\\a\\b\\c\\d.txt'#双\\指的是windows中的目录
# # print(file_path.split('\\',1))#这个是从右边以第一个\\为分隔符1))#这个取出的是以第一个\\为分隔符
# # print(file_path.rsplit('\\',))
# import numpy as np
# a =2**8+40
# b = -256
# c = 2**16-1
#
# a = np.int8(a)#0-255的格式，不管整数的正负，这里的a就是40
# b = np.uint8(b)#需要管正负，0-255，-1就是255，-256就是0
# c  = np.uint16(c)#0-2**16的范围
# d = np.int16(c)
# print(a)
# print(b)
# print(c)
# print(d)
# # 40
# # 0
# # 65535
# # # -1
# # print(2**16)
# a = [1,2,3,4,5]
# b = [6,7,8,9,2]
# for k, (i ,j) in zip(a,b):
#     print(i,j)
import PIL.ImageDraw as Draw
import PIL.Image as pimg
import matplotlib.pyplot as plt
import os
if __name__ == '__main__':
    path = r"D:\AIpractise\Projects\20190822_MTCNN\test_image"
    # detector = Detector()
    for image in os.listdir(path):
        print(os.listdir(path))
        print(image)
        image_path = os.path.join(path,image)
        print(image_path)
        img1 = pimg.open(image_path)
        plt.imshow(img1)
        plt.show()
        plt.pause(0.5)
        plt.close()

            # boxes = detector.detect(img1)
            # draw_img1 = Draw.ImageDraw(img1)
            # count = str(image_path[-6:-4])
            # print(count)
            #     # for box in boxes:
            # #
            # #     x1 = int(box[0])
            # #     y1 = int(box[1])
            # #     x2 = int(box[2])
            # #     y2 = int(box[3])
            # #     # print(box[4])
            # #     draw_img1.rectangle((x1,y1,x2,y2),outline="red",width=2)
            # img1.save(r"test_result\test{}.jpg".format(count))
            # plt.imshow(img1)
            # plt.show()
            # plt.pause(5)