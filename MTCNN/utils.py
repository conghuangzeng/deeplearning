import numpy as np

def iou(box,boxes,isMin=False):
    box_s=(box[2]-box[0])*(box[3]-box[1])
    boxes_s=(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    x1=np.maximum(box[0],boxes[:,0])
    y1 = np.maximum(box[1], boxes[:,1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
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
    _boxes=boxes[(-boxes[:,4]).argsort()]
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

# if __name__ == '__main__':

    # num=np.array([[ 75.4   ,99.5  ,284.29,372.25],[115 , 10 ,377, 272]])
    # print(nms(num,isMin=False))
    # a = np.array([98.5, 76.25, 313.85, 355.75]
    #              )
    # b = np.array([[34, 204, 244, 414]])
    # print(iou(a, b))

