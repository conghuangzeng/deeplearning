import os
import  PIL.Image as pimg
import PIL.ImageDraw as draw
import matplotlib.pyplot as plt
import utils
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

    a = open(sample_txt_path)

# negative_count = 1

count = 1

while True:
    if count>650288:
        break
    strs = "negative\{0}.jpg {1} {2} {3} {4} {5}\n".format(str(count), str(0), str(0), str(0), str(0),str(0))
    negative_sample_txt.write(strs)
    negative_sample_txt.flush()
    negative_sample_txt_24.write("negative\{0}.jpg {1} {2} {3} {4} {5}\n".format(str(count), str(0),str(0),str(0),str(0),str(0)))
    negative_sample_txt_24.flush()
    negative_sample_txt_12.write("negative\{0}.jpg {1} {2} {3} {4} {5}\n".format(str(count), str(0), str(0), str(0),str(0), str(0)))
    negative_sample_txt_12.flush()
    count += 1

print("数据生成完成")





