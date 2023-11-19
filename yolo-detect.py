import yolov5 
import cv2
import numpy
import pandas 
from sort import *
from process import Process
from configparser import ConfigParser



"""------------------- READ CONFIG -------------------"""

conf = ConfigParser()
conf.read("project.ini")

# model
model_vehicle_detect = conf.get("load_model","model_vehicle_detect")
model_plate_detect = conf.get("load_model","model_plate_detect")
model_plate_number_detect = conf.get("load_model","model_plate_number_detect")

# video
video = conf.get("load_video","video")

#mask
mask_path = conf.get("load_mask","mask")

# line separate
linex1 = conf.getint("init_line","linex1")
liney1 = conf.getint("init_line","liney1")
linex2 = conf.getint("init_line","linex2")
liney2 = conf.getint("init_line","liney2")

# destination of result images
save_image_path = conf.get("destination","save_image")

# get graphic setting
show_vehicle_detect = conf.getboolean("graphic_setting","show_vehicle_detect")
show_tracking = conf.getboolean("graphic_setting","show_tracking")


"""--------------------- END --------------------------"""

#load model vehicle detect
model = yolov5.load(model_vehicle_detect)


cap = cv2.VideoCapture(video)

mask = cv2.imread(mask_path)

# the separate line
line = [linex1, liney1, linex2, liney2]

class_names = ['moto', 'car', 'truck', 'bus', 'bycycle']


alpha = 20 # distance from the line to the object

#tracking deepsort
Tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.5)

track_list = [] # tracking list that to append obj id in list

# Processor
P = Process(save_image_path)
P.load_number_plate_picture(model_plate_detect)
P.load_number_plate(model_plate_number_detect)



# loop in video
while True:
    success, img = cap.read()
    masked_img = cv2.bitwise_and(img,mask)

    rs = model(masked_img) # use model

    predictions = rs.pred[0]  

    detections = np.empty((0,5)) # init detections for tracking - 5th index is class

    list_names = [] # init list names for tracking
    # get boxes, scores, class name
    for p in predictions:
        score = round(float(p[4]),2)
        x1,y1,x2,y2 = p[:4]
        x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
        if show_vehicle_detect:
            name = class_names[int(p[5])]
            list_names.append(int(p[5]))
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),2)

            cv2.line(img,(line[0],line[1]),(line[2],line[3]),color=(255,0,0),thickness=2) # draw the line

            cv2.putText(img,f'{name}-{score}',(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2) # draw name and score
        
        currentArray = np.array([x1,y1,x2,y2,int(p[5])]) #tracking

        detections = np.vstack((detections,currentArray)) # tracking

    tracker_rs = Tracker.update(detections)  # update 
    
    # get tracking boxes and draw
    for rs in tracker_rs:
        # ind = (len(list_names)-1) - i
        # print(rs," ", class_names[list_names[ind]])
        x1,y1,x2,y2,id = rs
        x1,y1,x2,y2,id = int(x1), int(y1), int(x2), int(y2), int(id)
        xc = (x2+x1) // 2
        yc = (y2+y1) // 2

        if show_tracking:
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2) # tracking boxes
            cv2.putText(img,f'{int(id)}',(x1+40,max(0,y1-10)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2) # draw tracking id

        cv2.circle(img,(xc,yc),2,(255,0,255),cv2.FILLED) # draw center of the object's box
        
        # append in tracking
        if line[2] - alpha < xc < line[0]+alpha and id not in track_list:
            track_list.append(id)
            print(track_list)
            # Pipeline processing
            new_img = P.save_image(img,id,x1,y1,x2,y2) # save image
            img_dect = P.plate_detection(new_img)
            P.number_plate_extract(img_dect)
            # end pipeline


    # print()
    # cv2.imshow("Masked-Image",masked_img)
    cv2.imshow("Image",img)
    cv2.waitKey(0)