
IMG_HEIGHT = 416
IMG_WIDTH = 416

img_h = 416
img_w = 416

CLASS_NUM = 10
#搞成字典的额形式更好用
ANCHORS_GROUP = {
	13:[[116,90],[156,198],[373,326]],
	26:[[30,61],[62,45],[59,119]],
	52:[[10,13],[16,30],[33,23]]}

anchor_area_groups = {
	13:[x*y for x,y in ANCHORS_GROUP[13]],
	26:[x*y for x,y in ANCHORS_GROUP[26]],
	52:[x*y for x,y in ANCHORS_GROUP[52]]}

