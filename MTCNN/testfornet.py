from nets import  Rnet
from torchvision import  transforms
import PIL.Image as pimg
import PIL.ImageDraw as Draw

net = Rnet()

transform = transforms.Compose([
    transforms.Resize((24,24)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])
img = pimg.open("test_image/1.jpg")
imgdata = transform(img).reshape(1,3,24,24)

_,zb = net(imgdata)
draw = Draw.ImageDraw(img)
w,x,y,z =0.11228070175438597,-0.017543859649122806, -0.09473684210526316,0.08070175438596491
draw.rectangle((zb[0,0]*24,zb[0,1]*24,zb[0,2]*24+24,zb[0,3]*24+24),outline="red")
draw.rectangle((w*24,x*24,y*24+24,z*24+24),outline="blue")

img.show()