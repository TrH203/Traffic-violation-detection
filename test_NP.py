import cv2
import numpy as np
from process import Process
import yolov5
from tool import rotate_and_crop
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def convert_to_list(l):
    for i in range(len(l) - 1):
        for j in range(i,len(l)):
            if l[i][2] > l[j][2]:
                l[i],l[j] = l[j],l[i]

    print(l)
    for i in range(4-1):
        for j in range(i+1,4):
            if l[i][1] > l[j][1]:
                l[i],l[j] = l[j],l[i]

    print(l)
    for i in range(4,len(l)-1):
        for j in range(i+1,len(l)):
            if l[i][1] > l[j][1]:
                l[i],l[j] = l[j],l[i]

    rs = ""
    for i in l:
        rs += i[0]

    return rs

image1 = cv2.imread("/Users/trHien/Python/MyProjects/YoloDetect/myenv/yolo-detect/plateImage2.png",cv2.IMREAD_COLOR)  
image1 = rotate_and_crop(image1)
image = image1.copy()
# image1 = cv2.GaussianBlur(image1, (15, 15), 0)
image1 = cv2.resize(image1,(224,224))
#image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)

class_names = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','K','L','M','N','P',
             'S','T','U','V','X','Y','Z','0']

# P1 = Process("/Users/trHien/Python/MyProjects/YoloDetect/myenv/Plate-Detect/License-Plate-Recognition/model/LP_detector.pt")
# P2 = Process("/Users/trHien/Python/MyProjects/YoloDetect/myenv/Plate-Detect/License-Plate-Recognition/model/LP_ocr.pt")
# print(image1.shape)
# P1.plate_detection(image1)
model = yolov5.load("/Users/trHien/Downloads/number_plate2/best.pt")
rs = model(image1) # use model

predictions = rs.pred[0]  

detections = np.empty((0,5)) # init detections for tracking - 5th index is class

l = []
# get boxes, scores, class name
for p in predictions:
    score = round(float(p[4]),2)
    x1,y1,x2,y2 = p[:4]
    x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
    # print(x1,y1,x2,y2)

    name = class_names[int(p[5])]
    print(name,score,(x1,y1,x2,y2))
    l.append([name,x1,y1,x2,y2])

    cv2.rectangle(image1,(x1,y1),(x2,y2),(255,0,255),2)

    # cv2.line(img,(line[0],line[1]),(line[2],line[3]),color=(255,0,0),thickness=2) # draw the line

    cv2.putText(image1,f'{name}',(x1,max(30,y1)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

try:
    rs = convert_to_list(l)
    pass
except:
    rs = "None"
print(rs)
cv2.imshow("image Predict",image1)
cv2.imshow("image Orginal",image)
cv2.waitKey(0)
cv2.destroyAllWindows()


