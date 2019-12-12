#播放视频
from detect_numpy import Detector
detector  = Detector()
import cv2
path = r"C:\Users\admin\Desktop\1.mp4"
cap = cv2.VideoCapture(path)#打开视频文件
print(cap)
fps = cap.get(cv2.CAP_PROP_FPS)#FPS帧数，帧数指每一秒有多少张图片
print(fps)
while True:
    ret,fram = cap.read()#这里ret指的是一种标识，把视频看成是图片的集合，相当于是不断的连续取图片，然后再以一定的时间间隔将图片显示出来。能取到图片ret就是True，然后以50ms的间隔输出图片内容fram，就是视频，当ret取不到东西了，while循环终止
    # print(ret)#bool值
    # print(fram)#内容
    if ret:#放完了ret就变成了Flase
        # print(ret)
        boxes = detector.detect(fram)
        for box in boxes:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            # print(box[4])
            cv2.rectangle((x1,y1),(x2,y2),color="red")
        cv2.imshow("honey",fram)#opencv用imshow显示出来
        #"honey"给当前视频文件一个名称，fram指图片内容，是numpy数组
        cv2.waitKey(50)#等待时间，毫秒
    else:
        break
cap.release()#释放cap视频文件
cv2.destroyAllWindows()#清除所有窗口的东西