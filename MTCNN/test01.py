import PIL.Image as pimg
import  PIL.ImageDraw as draw
import matplotlib.pyplot as plt

img  = pimg.open(r"D:\celeba1\shujuji\img_celeba\019969.jpg")
# img.show()

# txt =open(r"D:\celeba1\shujuji\list_bbox_celeba.txt","r")
# # print(txt.readlines())
# for  i,line in enumerate(txt):
#     print(line)

img_draw =  draw.ImageDraw(img)
img_draw.rectangle((197 ,444 , 197+47 ,444+ 65),outline="red",width=5)
img.show()
