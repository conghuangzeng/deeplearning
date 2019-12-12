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
    num1 = np.array( [[  0.6324543 , 272.98615   , 322.92703   ,  95.27891  ,   65.12056   ],
 [  0.6082531 , 284.20456  ,  325.57162  ,   45.029408  , 101.11346   ],
 [  0.63620996, 284.2416   ,  325.51828    , 44.719826 ,  107.30313   ],
 [  0.652427  , 168.78821  ,  340.66968   ,  57.528393  ,  65.076645  ],
 [  0.6797207 , 186.5753   ,  341.62704  ,   56.104076  ,  67.1678    ],
 [  0.643572  , 203.38531  ,  340.9363    ,  59.99302   ,  66.06185   ],
 [  0.6313416 , 274.00424  ,  340.54697   ,  67.430954  ,  65.15515   ],
 [  0.6198839 ,  40.684334 ,  377.66364   ,  55.150898  ,  70.03196   ],
 [  0.6491158 , 433.69586   , 378.31467  ,   55.45625   ,  62.418835  ],
 [  0.7722946 , 455.1891   ,  380.2555   ,   46.169647  ,  71.93367   ]])
    b = nms(num1,i=0.1,isMin=False)
    print(b)
