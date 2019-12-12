import os
import PIL.Image as pimg
import numpy as np
import utils #导入的iou和nms
import traceback#？？？

anno_src = r"D:\AIpractise\MTCNN\Anno\list_bbox_celeba.txt"
img_dir = r"D:\AIpractise\MTCNN\img_celeba"

save_path = r"D:\AIpractise\MTCNN\celeba4"

for face_size in [12]:#列表里放12,24,48三种尺寸的

    print("gen % i image" % face_size)
    # 样本图片存储路径
    positive_image_dir = os.path.join(save_path, str(face_size), "positive")#注意，三个都是文件夹，
    negative_image_dir = os.path.join(save_path, str(face_size), "negative")
    part_image_dir = os.path.join(save_path, str(face_size), "part")

    for dir_path in [positive_image_dir, negative_image_dir, part_image_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 样本描述存储路径，标签
    positive_anno_filename = os.path.join(save_path, str(face_size), "positive.txt")
    negative_anno_filename = os.path.join(save_path, str(face_size), "negative.txt")
    part_anno_filename = os.path.join(save_path, str(face_size), "part.txt")

    positive_count = 0
    negative_count = 0
    part_count = 0

    try:
        positive_anno_file = open(positive_anno_filename, "w")#
        negative_anno_file = open(negative_anno_filename, "w")
        part_anno_file = open(part_anno_filename, "w")

        for i, line in enumerate(open(anno_src)):
            if i < 2:#选取0 和1的样本
                continue
            try:
                strs = line.strip().split(" ")
                strs = list(filter(bool, strs))
                #strs = line.strip().spilt()直接用这个试试
                image_filename = strs[0].strip()
                print(image_filename)
                image_file = os.path.join(img_dir, image_filename)#生成文件路径

                with pimg.open(image_file) as img:
                    img_w, img_h = img.size#得到图片的尺寸
                    x1 = float(strs[1].strip())
                    y1 = float(strs[2].strip())
                    w = float(strs[3].strip())
                    h = float(strs[4].strip())#标签中索引为3和4的是框的宽和高，理解，重点注意
                    x2 = float(x1 + w)
                    y2 = float(y1 + h)

                    px1 = 0#float(strs[5].strip())
                    py1 = 0#float(strs[6].strip())
                    px2 = 0#float(strs[7].strip())
                    py2 = 0#float(strs[8].strip())
                    px3 = 0#float(strs[9].strip())
                    py3 = 0#float(strs[10].strip())
                    px4 = 0#float(strs[11].strip())
                    py4 = 0#float(strs[12].strip())
                    px5 = 0#float(strs[13].strip())
                    py5 = 0#float(strs[14].strip())

                    if max(w, h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                        #max(w, h) < 40人脸大小针对48而言最大不能超过40，x1 < 0 or y1 < 0说明有可能是半张脸，且处于图片左边的位置，w < 0 or h < 0书说明没有框。没有人脸？？？？？x1，y1大于0，x2，y2大于0也有可能是半张脸，处于图片右边的位置，这个需要根据iou排除
                        continue

                    boxes = [[x1, y1, x2, y2]]##原本的框的4个坐标值

                    # 计算出人脸中心点位置
                    cx = x1 + w / 2
                    cy = y1 + h / 2

                    # 使正样本和部分样本数量翻倍
                    for _ in range(5):#翻5倍
                        # 让人脸中心点有少许的偏移
                        w_ = np.random.randint(-w * 0.2, w * 0.2)
                        h_ = np.random.randint(-h * 0.2, h * 0.2)
                        cx_ = cx + w_
                        cy_ = cy + h_

                        # 让人脸形成正方形，并且让坐标也有少许的偏离
                        side_len = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
                        x1_ = np.max(cx_ - side_len / 2, 0)
                        y1_ = np.max(cy_ - side_len / 2, 0)#side_len / 2这个是人脸的中心点， cy_这个是让人脸中心有些许偏移的中心点
                        x2_ = x1_ + side_len
                        y2_ = y1_ + side_len

                        crop_box = np.array([x1_, y1_, x2_, y2_])#新的框

                        # 计算坐标的偏移值
                        offset_x1 = (x1 - x1_) / side_len#人脸的框是方的
                        offset_y1 = (y1 - y1_) / side_len#4个偏移量最后写入了标签txt文件中
                        offset_x2 = (x2 - x2_) / side_len
                        offset_y2 = (y2 - y2_) / side_len

                        offset_px1 = 0#(px1 - x1_) / side_len
                        offset_py1 = 0#(py1 - y1_) / side_len
                        offset_px2 = 0#(px2 - x1_) / side_len
                        offset_py2 = 0#(py2 - y1_) / side_len
                        offset_px3 = 0#(px3 - x1_) / side_len
                        offset_py3 = 0#(py3 - y1_) / side_len
                        offset_px4 = 0#(px4 - x1_) / side_len
                        offset_py4 = 0#(py4 - y1_) / side_len
                        offset_px5 = 0#(px5 - x1_) / side_len
                        offset_py5 = 0#(py5 - y1_) / side_len

                        # 剪切下图片，并进行大小缩放
                        face_crop = img.crop(crop_box)
                        face_resize = face_crop.resize((face_size, face_size))

                        iou = utils.iou(crop_box, np.array(boxes))[0]#索引为啥要取0？？？？
                        if iou > 0.65:  # 正样本,iou的数值需要调，不然效果不好。
                            positive_anno_file.write(
                                "positive/{0}.jpg {1} {2} {3} {4} {5}".format(
                                    positive_count, 1, offset_x1, offset_y1,
                                    offset_x2))
                            positive_anno_file.flush()#flush() 方法是用来刷新缓冲区的，即将缓冲区中的数据立刻写入文件，同时清空缓冲区，不需要是被动的等待输出缓冲区写入
                            face_resize.save(os.path.join(positive_image_dir, "{0}.jpg".format(positive_count)))
                            positive_count += 1
                        elif iou > 0.4:  # 部分样本
                            part_anno_file.write(
                                "part/{0}.jpg {1} {2} {3} {4} {5}".format(
                                    part_count, 2, offset_x1, offset_y1,offset_x2,
                                    offset_y2))
                            part_anno_file.flush()
                            face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                            part_count += 1
                        elif iou < 0.3:
                            negative_anno_file.write(
                                "negative/{0}.jpg {1} 0 0 0 0 0".format(negative_count, 0))
                            negative_anno_file.flush()#????
                            face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                            negative_count += 1

                        # 生成负样本，负样本多搞一些,负样本直接爬背景图好些
                        _boxes = np.array(boxes)

                    for i in range(5):#这种方法会扣到一部分的人脸，不好，直接去爬背景色复杂的没人脸的图片搞
                        side_len = np.random.randint(face_size, min(img_w, img_h) / 2)
                        x_ = np.random.randint(0, img_w - side_len)
                        y_ = np.random.randint(0, img_h - side_len)
                        crop_box = np.array([x_, y_, x_ + side_len, y_ + side_len])

                        if np.max(iou.iou(crop_box, _boxes)) < 0.3:
                            face_crop = img.crop(crop_box)
                            face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)

                            negative_anno_file.write("negative/{0}.jpg {1} 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(negative_count, 0))
                            negative_anno_file.flush()
                            face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                            negative_count += 1
            except Exception as e:
                traceback.print_exc()


    finally:
        positive_anno_file.close()
        negative_anno_file.close()
        part_anno_file.close()
