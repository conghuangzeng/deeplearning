import PIL.Image as pimg
import os

old_dir = r"D:\shujuji\cut144-simi1"
dir  = r"D:\shujuji\人脸识别训练集"
dir1 = os.listdir(old_dir)
# print(len(dir1))
#训练集目录下创建文件夹
nowpath = dir
print("现在的路径为：",dir)
# fileName = u"111"
# for  k  in range(3000):
#     fileName = str(k)
#     fileNamePath = nowpath+"\\"+fileName
#     os.makedirs(fileNamePath)


for i in range (len(dir1)):
    if i>3000:
        break
    for j,file in enumerate(os.listdir(os.path.join(old_dir,dir1[i]))):
        if j >10:
            break
        img_path = os.path.join(os.path.join(old_dir,dir1[i]),file)
        # print(img_path)
        img  = pimg.open(img_path)
        # img.show()
        img = img.resize((96,96))
        img.save("{0}/{1}.jpg".format(os.path.join(dir,str(i)),j))
    print(i)
print("完成")

