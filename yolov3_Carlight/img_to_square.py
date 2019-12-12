import PIL.Image as pimg

img= pimg.open(r"test_img/1.jpg")
w,h = img.size
print(w,h)

img = img.crop((8,8,424,424))
w,h = img.size
print(w,h)
# img.show()
# img.save(r"dataset_data/1.jpg")