import PIL.Image as pimg

path = r"C:\Users\Admin\Desktop\imgdata\image（初减）"
import os
count = 1
for image in os.listdir(path):
	print(image)
	img = pimg.open(os.path.join(path,image))
	img  = img.convert("RGB")
	img1 = img.resize((416,416))
	img1.save(r"C:\Users\Admin\Desktop\imgdata\{0}.jpg".format(count))
	count+=1
