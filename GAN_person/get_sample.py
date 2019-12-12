import os
import PIL.Image as pimg
DIR = r"D:\shujuji\img_align_celeba"
img_path  = r"D:\shujuji\人脸图像生成GAN"
count = 1
for  i ,file  in enumerate(os.listdir(DIR)):
    if i >50000:
        break
    img = pimg.open(os.path.join(DIR,file))
    img = img.resize((96,96))
    # img = img.resize((100,100))
    img.save("{0}\{1}.jpg".format(img_path,count))
    count+=1
    print(count)
