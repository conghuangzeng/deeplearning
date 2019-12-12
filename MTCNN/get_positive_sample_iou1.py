import os
import  PIL.Image as pimg
import PIL.ImageDraw as draw
import matplotlib.pyplot as plt
import random
import  numpy as np
import utils
import traceback
total_DIR = r"D:\celeba"
sample_txt_path = r"D:\celeba\anno_landmark.txt"
image_DIR = r"D:\celeba1\shujuji\img_celeba"
list = [48]
for  size in list:
    positive_sample_path = os.path.join(total_DIR,str(size),"positive")
    part_sample_path = os.path.join(total_DIR, str(size), "part")
    negative_sample_path = os.path.join(total_DIR, str(size), "negative")
    for file in [positive_sample_path,part_sample_path,negative_sample_path]:
        if not os.path.exists(file):
            os.makedirs(file)
    positive_sample_txt_path = os.path.join(total_DIR, str(size),"positive.txt")
    # part_sample_txt_path = os.path.join(total_DIR, str(size), "part.txt")
    # negative_sample_txt_path = os.path.join(total_DIR, str(size), "negative.txt")


    positive_sample_txt = open(positive_sample_txt_path,mode="w")
    # part_sample_txt = open(positive_sample_txt_path,"w")
    # negative_sample_txt = open(negative_sample_txt_path,"w")

    positive_sample_txt_24 = open(os.path.join(total_DIR, str(24), "positive.txt"), mode="w")
    positive_sample_txt_12 = open(os.path.join(total_DIR, str(12), "positive.txt"), mode="w")
    a = open(sample_txt_path)
    print(a)
    positive_count = 1
    part_count = 1
    negative_count = 1
    while True:
        # print(1)
        a = open(sample_txt_path)
        for i, line in enumerate(a):  # 枚举
            # print(line)
            try:

                if i < 2:
                    continue
                if positive_count > 200000:
                    break
                line = line.strip().split()
                # print(line)
                # print(line[0])
                #五官位置
                x_1 = min(int(line[5]),int(line[7]))
                y_1 =min(int(line[6]),int(line[8]))
                x_2=max(int(line[11]),int(line[13]))
                y_2=max(int(line[12]),int(line[14]))
                #原框
                x11 = int(line[1])
                y11 = int(line[2])
                w = int(line[3])
                h = int(line[4])
                x22 = int(x11 + w)
                y22 = int(y11 + h)
                #调整后的框
                x1 = x_1 + (x11 - x_1) * 0.95
                y1 = y_1 + (y11 - y_1) * 0.95
                x2 = x_2 + (x22 - x_2) * 0.87
                y2 = y_2 + (y22 - y_2) * 0.8

                _w = x2 - x1
                _h = y2 - y1
                sample_image_path = os.path.join(image_DIR, line[0])
                img = pimg.open(sample_image_path)
                count = 0
                w_img, h_img = img.size
                while count<2:

                    count += 1
                    # w_image, h_image = img.size
                    cx = (x2+x1)/2#中心点坐标
                    cy = (y2+y1)/2

                    max_side_len = max(_w,_h)
                #搞成正方形
                    try:
                        offset_c_x = random.randint(int(-0.1 * _w), int(0.1 * _w) + 2)
                        offset_c_y = random.randint(int(-0.1 * _h), int(0.1 * _h) + 2)
                        side_len = np.random.randint(0.9 * min(_w, _h), 1.1 * max(_w, _h))

                        c_x = cx + offset_c_x
                        c_y = cy + offset_c_y

                        _x1 = int(c_x - side_len / 2)
                        _y1 = int(c_y - side_len / 2)
                        _x2 = int(_x1 + side_len)
                        _y2 = int(_y1 + side_len)
                        # print(_x1,_y1,_x2,_y2)
                        # 计算偏移量,用最原始的框计算
                        offset_x1 = (x11 - _x1) / side_len
                        offset_y1 = (y11 - _y1) / side_len
                        offset_x2 = (x22 - _x2) / side_len
                        offset_y2 = (y22 - _y2) / side_len
                        if _x1 > 0 and _y1 > 0 and _x2 < w_img and _y2 < h_img and _x1 < _x2 and _y2 > _y1:
                            box1 = np.array([[_x1, _y1, _x2, _y2]])
                            box = np.array([x11, y11, x22, y22])
                            iou = utils.iou(box, box1, isMin=False)
                    # print(iou)

                            if iou > 0.85 :
                                draw_img = draw.ImageDraw(img)
                                draw_img.rectangle((box1[0, 0], box1[0, 1], box1[0, 2], box1[0, 3]), outline="green", width=5)
                                # #
                                draw_img.rectangle((x11, y11, x22, y22), outline="red", width=5)
                                plt.imshow(img)
                                plt.pause(1)
                    #     # 保存图片，写标签

            #                     img1 = img.crop((x1,y1,x2,y2))
            #                     img1 = img1.resize((48,48))
            #                     positive_sample_txt.write("positive\{0}.jpg {1} {2} {3} {4} {5}\n".format(str(positive_count),str(1),str(offset_x1),str(offset_y1),str(offset_x2),str(offset_y2)))
            #                     img1.save("{0}\{1}.jpg".format(os.path.join(total_DIR, str(48), "positive"),str(positive_count)))
            #                     positive_sample_txt.flush()
            #
            #                     #生成24图片
            #                     img1 = img1.resize((24,24))
            #                     positive_sample_txt_24.write(
            #                         "positive\{0}.jpg {1} {2} {3} {4} {5}\n".format(str(positive_count), str(1), str(offset_x1),str(offset_y1),str(offset_x2),str(offset_y2)))
            #                     img1.save("{0}\{1}.jpg".format(os.path.join(total_DIR, str(24), "positive"), str(positive_count)))
            #                     positive_sample_txt_24.flush()
            #
            #                     #生成12图片
            #                     img1 = img1.resize((12,12))
            #                     positive_sample_txt_12.write(
            #                         "positive\{0}.jpg {1} {2} {3} {4} {5}\n".format(str(positive_count), str(1), str(offset_x1),str(offset_y1),str(offset_x2),str(offset_y2)))
            #                     img1.save("{0}\{1}.jpg".format(os.path.join(total_DIR, str(12), "positive"), str(positive_count)))
            #                     positive_sample_txt_12.flush()
            #                     positive_count += 1
            #                     print(positive_count)
            #                 else:
            #                     continue
                    except Exception as e:
                        traceback.print_exc()

            except Exception as e:
                traceback.print_exc()
            # print("生成数据完成")




