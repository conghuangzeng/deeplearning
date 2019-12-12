import  PIL.Image as pimg
import  numpy as np

file  =pimg.open(r"F:\面试\头像.jpg")
img = np.array(file,dtype=np.uint8)
pixes = file.getpixel((75,8))
# file.show()
print(pixes)
img1 = pimg.new(mode= "RGB", size = (500,500), color=(0, 101, 149))
# img1.show()
# img1.paste(file,(200,200))
img1.save("./1.jpg")
img1.show()
