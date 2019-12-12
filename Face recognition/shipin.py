#播放，存储视频
import cv2
from detect_shipin import Detector

path = r"C:\Users\admin\Desktop\1.mp4"
cap = cv2.VideoCapture(path)#打开视频文件
# detector  = Detector()
fps = cap.get(cv2.CAP_PROP_FPS)#FPS帧数，帧数指每一秒有多少张图片
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 保存视频的编码
# out = cv2.VideoWriter('7.avi', fourcc, 20.0, (int(cap.get(3)),int(cap.get(4))))
out = cv2.VideoWriter('7.avi', fourcc, 20.0, (int(416),int(416)))
count = 0
i=0
boxes=""
# detect = Detector()
label_count = 1
while True:
    ret,fram = cap.read()
    if ret:#ret是bool值，理解
        # _image = cv2.cvtColor(fram, cv2.COLOR_BGR2RGB)#必须要转换，不然传到网络里面去精度太低了。
        _image = fram[10:426, 200:616]
        image = fram
        i+=1

        # print(fram.shape)
        # cv2.imshow("show", fram)
        # print(_image.shape)

        # if count % 4 == 0 and count > 0:
            # boxes = detect.main(_image)
            # print(boxes)
        # for box in boxes:
        #     x1 = int(box[1])
        #     y1 = int(box[2])
        #     x2 = int(box[3])
        #     y2 = int(box[4])
        #         # rec =
        #         # print
        #     text = "Brake-light on"
            # cv2.putText(image, text, (x1,y1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,255,255), 1)
            # cv2.rectangle(image, (x1, y1), (x2, y2), (0,255, 255))
        # cv2.imwrite("test_img/{0}.jpg".format(label_count),image)
        label_count += 1
        cv2.imshow("show",_image)#原图上画框，show的也是原图
        # print(image.shape)
        cv2.waitKey(30)#等待时间，毫秒
        # if i < 800:

        out.write(_image)#存储每一帧的内容
        count += 1
        i+=1

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    else:
        break
cap.release()#释放cap视频文件
cv2.destroyAllWindows()#清除所有窗口的东西