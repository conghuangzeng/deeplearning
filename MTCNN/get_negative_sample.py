import os
import  PIL.Image as pimg
import PIL.ImageDraw as draw
import matplotlib.pyplot as plt
import utils
import numpy as np
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
    part_sample_txt_path = os.path.join(total_DIR, str(size), "part.txt")
    negative_sample_txt_path = os.path.join(total_DIR, str(size), "negative.txt")

    print(negative_sample_txt_path)
    negative_sample_txt = open(negative_sample_txt_path,mode="w")
    negative_sample_txt_12 =open(os.path.join(total_DIR, str(12), "negative.txt"),mode="w")
    negative_sample_txt_24 = open(os.path.join(total_DIR, str(24), "negative.txt"), mode="w")

    a = open(sample_txt_path)

    negative_count =1

    while True:
        # print(1)
        a = open(sample_txt_path)
        for  i,line in enumerate(a):#枚举
                # print(line)
            try:

                if i<2:
                    continue
                if negative_count>600000:
                    break

                line = line.strip().split()
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

                sample_image_path = os.path.join(image_DIR, line[0])
                img = pimg.open(sample_image_path)
                draw_img = draw.ImageDraw(img)
                count= 0
                _w = x22 - x11
                _h = y22 - y11
                count = 0
                w_img, h_img = img.size
                try:
                    cx = (x22 + x11) / 2  # 中心点坐标
                    cy = (y22 + y11) / 2
                    while count<3:
                        count+=1
                        offset_c_x = np.random.randint(-0.8*_w,  0.8*_w)
                        offset_c_y = np.random.randint(-0.8* _h, 0.8*_h)
                        side_len = np.random.randint(0.8 * min(w, h), 1.2 * max(w, h))

                        c_x = cx + offset_c_x
                        c_y = cy + offset_c_y

                        _x1 = int(c_x - side_len / 2)
                        _y1 = int(c_y - side_len / 2)
                        _x2 = int(_x1 + side_len)
                        _y2 = int(_y1 + side_len)

                        if _x1 > 0 and _y1 > 0 and _x2<w_img and _y2<h_img and c_x>0  and c_x<w_img and c_y>0  and c_x<h_img:

                            box1 = np.array([[_x1, _y1, _x2, _y2]])
                            box = np.array([x11, y11, x22, y22])
                            iou = utils.iou(box, box1, isMin=False)

                            if  iou < 0.1:
                                # draw_img = draw.ImageDraw(img)
                                # draw_img.rectangle((box1[0, 0], box1[0, 1], box1[0, 2], box1[0, 3]), outline="green", width=5)
                                # # #
                                # draw_img.rectangle((x11, y11, x22, y22), outline="red", width=5)
                                # plt.imshow(img)
                                # plt.pause(1)
                                #保存图片，写标签
                                img1 = img.crop((_x1,_y1,_x2, _y2))
                                img1 = img1.resize((48,48))
                                strs = "negative\{0}.jpg {1} {2} {3} {4} {5}\n".format(str(negative_count),str(0),str(0),str(0),str(0),str(0))

                                negative_sample_txt.write(strs)

                                negative_sample_txt.flush()
                                img1.save("{0}\{1}.jpg".format(os.path.join(total_DIR, str(size), "negative"),str(negative_count)))
                                img1 = img1.resize((24,24))
                                #
                                negative_sample_txt_24.write("negative\{0}.jpg {1} {2} {3} {4} {5}\n".format(str(negative_count), str(0),str(0),str(0),str(0),str(0)))
                                negative_sample_txt_24.flush()
                                img1.save("{0}\{1}.jpg".format(os.path.join(total_DIR, str(24), "negative"),str(negative_count)))
                                img1 = img1.resize((12,12))
                                negative_sample_txt_12.write(
                                    "negative\{0}.jpg {1} {2} {3} {4} {5}\n".format(str(negative_count), str(0), str(0), str(0),str(0), str(0)))
                                negative_sample_txt_12.flush()
                                img1.save("{0}\{1}.jpg".format(os.path.join(total_DIR, str(12), "negative"), str(negative_count)))
                                negative_count += 1
                                print(negative_count)
                            else:
                                continue
                        else:
                            continue

                except Exception as e:
                    traceback.print_exc()

            except Exception as e:
                traceback.print_exc()
            #
    print("数据生成完成")






