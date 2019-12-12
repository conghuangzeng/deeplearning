import PIL.Image as pimg
import os


path = r"./dataset/train_img"
count = 1
for  imgs in os.listdir(path):
    img = pimg.open(os.path.join(path,imgs))
    # img.show()
    img = img.convert("RGB")
    img.save("{0}.jpg".format(count))
    count+=1