import os
import  PIL.Image as pimg
import PIL.ImageDraw as draw
import matplotlib.pyplot as plt
import utils
import traceback
import numpy as np
total_DIR = r"D:\celeba"
sample_txt_path = r"D:\celeba\anno_landmark.txt"
image_DIR = r"D:\celeba1\shujuji\img_celeba"
list = [48]
for  size in list:

    negative_sample_path = os.path.join(total_DIR, str(size), "negative")

    negative_sample_txt_path = os.path.join(total_DIR, str(size), "negative.txt")

    print(negative_sample_txt_path)
    negative_sample_txt = open(negative_sample_txt_path,mode="w")
    negative_sample_txt_12 =open(os.path.join(total_DIR, str(12), "negative.txt"),mode="w")
    negative_sample_txt_24 = open(os.path.join(total_DIR, str(24), "negative.txt"), mode="w")

bg_img1 = r"D:\celeba1\pic1"

negative_count = 600001
while True:
    a = open(sample_txt_path)
    if negative_count>650000:
        break
    for name in os.listdir(bg_img1):#将bg_img1文件夹里的东西以列表的形式表现出来

        img= pimg.open(os.path.join(bg_img1,name))#打开图片

        img = img.convert("RGB")#

        w,h = img.size
        try:
            _x1 = int(np.random.randint(0,w*0.5))
            _y1 = int(np.random.randint(0,h*0.5))
            side_len =int(np.random.randint(0,max(w,h)))
            _x2 = int(_x1+side_len)
            _y2 = int(_y1+side_len)

            # if _x2<w  and _y2<h:
                #
                # strs = "negative\{0}.jpg {1} {2} {3} {4} {5}\n".format(str(negative_count), str(0), str(0), str(0), str(0))
                #
                # negative_sample_txt.write(strs)
                #
                # negative_sample_txt.flush()
                #
                # negative_sample_txt_24.write("negative\{0}.jpg {1} {2} {3} {4} {5}\n".format(str(negative_count), str(0),str(0),str(0),str(0),str(0)))
                # negative_sample_txt_24.flush()
                #
                # negative_sample_txt_12.write(
                #                         "negative\{0}.jpg {1} {2} {3} {4} {5}\n".format(str(negative_count), str(0), str(0), str(0),str(0), str(0)))
                # negative_sample_txt_12.flush()
                # img = img.crop((_x1, _y1, _x2, _y2))
                # img = img.resize((48, 48))
                # img.save("{0}\{1}.jpg".format(os.path.join(total_DIR, str(48), "negative"), str(negative_count)))
                # img = img.resize((24, 24))
                # img.save("{0}\{1}.jpg".format(os.path.join(total_DIR, str(24), "negative"), str(negative_count)))
                # img = img.resize((12, 12))
                # img.save("{0}\{1}.jpg".format(os.path.join(total_DIR, str(12), "negative"), str(negative_count)))
                # negative_count += 1
        except Exception as e:
            traceback.print_exc()
        print(negative_count)
    print("数据生成完成")





