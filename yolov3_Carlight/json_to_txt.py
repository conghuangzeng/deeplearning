import json
import os
path  = r"data-json/outputs"

# for i in range(10):
path1 = os.listdir(path)
# path1 = path.sort()
# print(path1)
# count = 1
for i ,files in enumerate(path1):
	# if i >2:
	# 	break
	print(files)#注意load和loads的区别
	file = open(os.path.join(path, files), encoding="utf-8")
	# print(file[0])
	# print(len(file))


	b = json.load(file)
	image_path = b["path"]

	print(image_path[-6:])
	# print(len(b["outputs"]["object"]))
	for j in range(len(b["outputs"]["object"])):


		cls = b["outputs"]["object"][j]["name"]
		# print(cls)
		num1  = b["outputs"]["object"][j]["bndbox"]
		# # cls_num =i
		x11 = num1["xmin"]
		y11 =num1["ymin"]
		x12 =num1["xmax"]
		y12 =num1["ymax"]
		cx = int((x11+x12)*0.5)
		cy = int((y11+y12)*0.5)
		w =int(x12 -x11)
		h  = int(y12-y11)
		with open(r"dataset_data/label.txt", mode="a") as  f:
			if len(b["outputs"]["object"])==1:
				f.write("{0}  {1}  {2}  {3}  {4}  {5}  \n".format(str(files), cls, cx, cy, w, h))
				break
			if j==0:
				f.write("{0}  {1}  {2}  {3}  {4}  {5}  ".format(str(files), cls, cx, cy, w, h))
			if j>0 and j <(len(b["outputs"]["object"])-1):
				f.write("{0}  {1}  {2}  {3}  {4}  ".format(cls, cx, cy, w, h))
			if j ==(len(b["outputs"]["object"])-1):

				f.write("{0}  {1}  {2}  {3}  {4}  \n".format(cls, cx, cy, w, h))
	# count += 1


