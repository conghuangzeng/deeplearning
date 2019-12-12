import os
import  PIL.Image as pimg
import PIL.ImageDraw as draw
import matplotlib.pyplot as plt
import numpy as np
import utils,random
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

    part_sample_txt_path = os.path.join(total_DIR, str(size), "part.txt")
    part_sample_txt_path_24 = os.path.join(total_DIR, str(24), "part.txt")
    part_sample_txt_path_12 = os.path.join(total_DIR, str(12), "part.txt")

    part_sample_txt = open(part_sample_txt_path,"w")
    part_sample_txt_24 = open(part_sample_txt_path_24, "w")
    part_sample_txt_12 = open(part_sample_txt_path_12, "w")




    part_count =1



    while True:
        # print(1)
        a = open(sample_txt_path)
        for  i,line in enumerate(a):#枚举
                # print(line)
            try:

                if i<2:
                    continue
                if part_count>200100:
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
                count = 0
                r_boxes = []
                #原框调整框后的坐标
                x1 = x_1 + (x11 - x_1) * 0.95
                y1 = y_1 + (y11 - y_1) * 0.95
                x2 = x_2 + (x22 - x_2) * 0.87
                y2 = y_2 + (y22 - y_2) * 0.8
                _w = x2 -x1
                _h  = y2 -y1

                count = 0
                w_img ,h_img = img.size
                while count<1:

                    count += 1
                    # w_image, h_image = img.size
                    cx = (x2+x1)/2#中心点坐标
                    cy = (y2+y1)/2

                    if _w ==0 or _h==0 :
                        print(_w,_h)
                        continue
                    try:
                        offset_c_x =random.randint(int(-0.4*_w),int(0.4*_w)+2)
                        offset_c_y = random.randint(int(-0.4*_h),int(0.4*_h)+2)
                        side_len = np.random.randint(0.9*min(_w,_h),1.2*max(_w,_h))


                        c_x = cx +  offset_c_x
                        c_y = cy + offset_c_y

                        _x1 = int(c_x - side_len / 2 )
                        _y1 = int(c_y - side_len / 2)
                        _x2 = int(_x1 + side_len)
                        _y2 = int(_y1 + side_len)
                        # print(_x1,_y1,_x2,_y2)
                        #计算偏移量,用调整后的框计算
                        offset_x1 = ( x1-_x1) / side_len
                        offset_y1 = ( y1-_y1 ) / side_len
                        offset_x2 = (  x2-_x2) / side_len
                        offset_y2 = ( y2-_y2 ) / side_len
                        if  _x1>0 and _y1 > 0 and _x2<w_img and _y2<h_img and _x1<_x2 and _y2>_y1:

                            box1 = np.array([[_x1,_y1,_x2,_y2]])
                            box = np.array([x1,y1,x2,y2])
                            iou = utils.iou(box,box1,isMin=False)
                            # print(iou)


                            if iou>0.35 and iou<0.5:



                                    # 保存图片，写标签
                                img1 = img.crop((_x1,_y1,_x2,_y2))
                                img1 = img1.resize((48,48))
                                part_sample_txt.write("part\{0}.jpg {1} {2} {3} {4} {5}\n".format(str(part_count),str(2),str(offset_x1),str(offset_y1),str(offset_x2),str(offset_y2)))
                                part_sample_txt.flush()

                                img1.save("{0}\{1}.jpg".format(os.path.join(total_DIR, str(48), "part"),str(part_count)))

                                    #生成24图片
                                img1 = img1.resize((24,24))
                                part_sample_txt_24.write(
                                        "part\{0}.jpg {1} {2} {3} {4} {5}\n".format(str(part_count), str(2), str(offset_x1),str(offset_y1),str(offset_x2),str(offset_y2)))
                                part_sample_txt_24.flush()

                                img1.save("{0}\{1}.jpg".format( os.path.join(total_DIR, str(24), "part"), str(part_count)))


                                    #生成12图片
                                img1 = img1.resize((12,12))
                                part_sample_txt_12.write(
                                "part\{0}.jpg {1} {2} {3} {4} {5}\n".format(str(part_count), str(2), str(offset_x1),str(offset_y1),str(offset_x2),str(offset_y2)))
                                part_sample_txt_12.flush()


                                img1.save("{0}\{1}.jpg".format(os.path.join(total_DIR, str(12), "part"), str(part_count)))
                                part_count+=1
                                print(part_count)
                            else:
                                continue
                        else:
                            continue
                    # continue
                    except Exception as e:
                        traceback.print_exc()
            except Exception as e:
                traceback.print_exc()


    print("数据生成完成")






