# 通过解析xml文件
'''
try:
    import xml.etree.CElementTree as ET
except:
    import xml.etree.ElementTree as ET

从Python3.3开始ElementTree模块会自动寻找可用的C库来加快速度
'''
import xml.etree.ElementTree as ET
import os
import sys


if __name__ == "__main__":
    xmls_path =r"D:\AIpractise\Projects\20190927_yolo_v3\data-xml"
    target_path = r"D:\AIpractise\Projects\20190927_yolo_v3\data-txt"

    for xmlFilePath in os.listdir(xmls_path):
        print(os.path.join(xmls_path,xmlFilePath))
        try:
            tree = ET.parse(os.path.join(xmls_path,xmlFilePath))

            # 获得根节点
            root = tree.getroot()
        except Exception as e:  # 捕获除与程序退出sys.exit()相关之外的所有异常
            print("parse test.xml fail!")
            sys.exit()

        # objects = root.find("object")
        # print(len(objects))
        f = open(target_path +"/" + os.path.splitext(xmlFilePath)[0] + ".txt", 'w')
        # print(f)

        for bndbox in root.iter('bndbox'):
            node = []
            for child in bndbox:
                node.append(int(child.text))
            x1, y1 = node[0],node[1]
            x2, y2 = node[2],node[3]
            # x2 ,y2 = x3 ,y1
            # x4, y4 = x1, y3
            # string = ''+str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+','+''+str(x3)+','+str(y3)+','+''+str(x4)+','+str(y4)+',test';
            string = '' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2)  + ',test';
            # print(string)
            f.write(string+'\n')
        f.close()