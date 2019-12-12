from PIL import Image,ImageDraw,ImageFont,ImageFilter
import random
import os

def ranChr():
    # a = chr(random.randint(48,57))
    # b = chr(random.randint(65,90))
    # c = chr(random.randint(97,122))
    return str(random.randint(0,9))
    # return chr(random.randint(48,57))
    # return random.choice([a,b,c])
    # return random.sample([a,b,c],1)[0]
def Color1():
    return (
        random.randint(64,255),
        random.randint(64,255),
        random.randint(64,255),
    )
def Color2():
    return (
        random.randint(32,128),
        random.randint(32,128),
        random.randint(32,128),
    )

font = ImageFont.truetype("msyhbd.ttf",40)
h=60
w=240
count =0
for i in range(10):
    img = Image.new("RGB",(w,h),(255,255,255))
    draw = ImageDraw.Draw(img)
    for x in range(w):
        for y in range(h):
            draw.point((x,y),fill=Color1())
    filename =""
    for j in range(4):
        char = ranChr()
        draw.text((60*j+10,5),char,font=font,fill=Color2())
        if j <3:

            filename+=char+"."
        if j==3:
            filename+=char
    if not os.path.exists("test_img"):
        os.makedirs("test_img")
    count +=1
    img.save("{0}/{1}.{2}.jpg".format("test_img",count,filename))
    print(i)