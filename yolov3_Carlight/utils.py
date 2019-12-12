import numpy as np

def iou(box,boxes,isMin=False):
    box_s=(box[4]-box[2])*(box[3]-box[1])
    boxes_s=(boxes[:,4]-boxes[:,2])*(boxes[:,3]-boxes[:,1])
    x1=np.maximum(box[1],boxes[:,1])
    y1 = np.maximum(box[2], boxes[:,2])
    x2 = np.minimum(box[3], boxes[:, 3])
    y2 = np.minimum(box[4], boxes[:, 4])
    w = np.maximum(0,x2-x1)
    h = np.maximum(0,y2-y1)
    inter = w* h
    if isMin:
        inter1 = np.true_divide(inter,np.minimum(box_s,boxes_s))

    else:
        inter1 = np.true_divide(inter,(box_s+boxes_s-inter))

    return inter1


def nms(boxes,i=0.3,isMin=False):
    if boxes.shape[0] <1:
        return np.array([])
    _boxes=boxes[(-boxes[:,0]).argsort()]
    # print(_boxes)
    r_boxes = []

    while _boxes.shape[0]>1:

        a_box = _boxes[0]
        b_boxes = _boxes[1:]
        r_boxes.append(a_box)
        index = np.where(iou(a_box,b_boxes,isMin)<i)

        _boxes= b_boxes[index]

        # r_boxes.append(_boxes)
    if _boxes.shape[0]>0:
        r_boxes.append(_boxes[0])

    nms_box = np.stack(r_boxes)
    # print(nms_box)
    return nms_box

if __name__ == '__main__':

    num=np.array([[ 0.8,75.4   ,99.5  ,284.29,372.25,1,2,3,4,5],[ 0.8,300   ,300  ,500.29,500.25,1,2,3,4,5],[0.7,115 , 10 ,377, 272,1,2,3,4,5]])
    num1 = np.array([[  0.7837696 ,178.71213  ,  92.152405 , 266.4518 ,   371.71863  ,   0.       ],
 [  0.5722121 ,186.10622   ,219.78589  , 275.68518  , 308.36548   ,  0.       ]])
    b = nms(num1,i=0.3,isMin=False)
    print(b)
    # index = np.where(b[:,0])
    # print(index)
    # # c = num [index,:][:,:,5:8]
    # c = num[index][:,0:6]
    # print(c)
    # a = np.array([98.5, 76.25, 313.85, 355.75]
    #              )
    # b = np.array([[34, 204, 244, 414]])
    # print(iou(a, b))

