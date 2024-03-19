import yolov5
import cv2

import numpy
import pandas
from sort import *
from process import Process
from configparser import ConfigParser
import ultralytics
from ultralytics import YOLO
"""------------------- READ CONFIG -------------------"""

conf = ConfigParser()
conf.read("project.ini")

# model
model_vehicle_detect = conf.get("load_model", "model_vehicle_detect")
model_plate_detect = conf.get("load_model", "model_plate_detect")
model_plate_number_detect = conf.get("load_model", "model_plate_number_detect")
model_helmet_detect = conf.get("load_model","model_helmet_detect")
# video
video = conf.get("load_video", "video")

# mask
mask_path = conf.get("load_mask", "mask")

# line separate
# linex1 = conf.getint("init_line", "linex1")
# liney1 = conf.getint("init_line", "liney1")
# linex2 = conf.getint("init_line", "linex2")
# liney2 = conf.getint("init_line", "liney2")

# destination of result images
save_image_path = conf.get("destination", "save_image")

# get graphic setting
show_vehicle_detect = conf.getboolean("graphic_setting", "show_vehicle_detect")
show_tracking = conf.getboolean("graphic_setting", "show_tracking")

"""--------------------- END --------------------------"""

# load model vehicle detect
model = yolov5.load(model_vehicle_detect)
model2 = YOLO(model_helmet_detect)
cap = cv2.VideoCapture(video)

mask = cv2.imread(mask_path)

# the separate line
# line = [linex1, liney1, linex2, liney2]

class_names = ['car', 'moto', 'truck', 'bus', 'bycycle']

alpha = 20  # distance from the line to the object

# tracking deepsort
Tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.5)

track_list = []  # tracking list that to append obj id in list
track_list2 = []
# Processor
P = Process(save_image_path)
P.load_number_plate_picture(model_plate_detect)
P.load_number_plate(model_plate_number_detect)


drawing = False
line_start = (-1, -1)
line_end = (-1, -1)
lines = []
line = [0,0,0,0]
def draw(event, x, y, flags, param):
    global drawing, line_start, lines, canvas, line_end, line

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        line_start = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        line_end = (x, y)
        lines.append([line_start[0], line_start[1], line_end[0], line_end[1]])
        cv2.line(canvas, line_start, line_end, (0, 255, 0), 2)
        print("Line drawn from ({}, {}) to ({}, {})".format(line_start[0], line_start[1], line_end[0], line_end[1]))
        line = [line_start[0], line_start[1], line_end[0], line_end[1]]
        return line
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', draw)
# loop in video
helmet = 0
while True:
    success, img = cap.read()
    masked_img = cv2.bitwise_and(img, mask)
    rs = model(masked_img)  # use model # vihecle detect
    rs2 = model2(masked_img) # helmet detect
    ano = rs2[0].plot() # plot helmet detect
    predictions = rs.pred[0]
    # print(line)
    detections = np.empty((0, 5))  # init detections for tracking - 5th index is class

    list_names = []  # init list names for tracking
    # get boxes, scores, class name
    for p in predictions:
        score = round(float(p[4]), 2)
        x1, y1, x2, y2 = p[:4]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if show_vehicle_detect:
            name = class_names[int(p[5])]
            list_names.append(int(p[5]))
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

            cv2.line(img, (line[0], line[1]), (line[2], line[3]), color=(255, 0, 0), thickness=2)  # draw the line

            cv2.putText(img, f'{name}-{score}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        2)  # draw name and score

        currentArray = np.array([x1, y1, x2, y2, int(p[5])])  # tracking
        detections = np.vstack((detections, currentArray))  # tracking
    tracker_rs = Tracker.update(detections)  # update

    # get tracking boxes and draw
    for rs in tracker_rs:
        # ind = (len(list_names)-1) - i
        # print(rs," ", class_names[list_names[ind]])
        x1, y1, x2, y2, id = rs
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        xc = (x2 + x1) // 2
        yc = (y2 + y1) // 2

        if show_tracking:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # tracking boxes
            cv2.putText(img, f'{int(id)}', (x1 + 40, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255),
                        2)  # draw tracking id

        # cv2.circle(img, (xc, yc), 2, (255, 0, 255), cv2.FILLED)  # draw center of the object's box

        # append in tracking
        if line[2] - alpha < xc < line[0] + alpha and id not in track_list:
            track_list.append(id)

            # Pipeline processing
            new_img = P.save_image(img, id, x1, y1, x2, y2, "image")  # save image
            img_dect = P.plate_detection(new_img)
            P.number_plate_extract(img_dect)
            # end pipeline

    # for result in rs2:
    #     boxes = result.boxes.cpu().numpy()
    #     conf = result.boxes.conf.numpy()
    #     r = boxes.xyxy
    #     class_label = boxes.cls
    #     for box, label, conf in zip(r, class_label, conf):
    #         if label == 0 and conf > 0.5:  # label 1 = no helmet, label 0 = helmet, conf = confidence of boxes
    #             # print(x1)
    #
    #             x1, y1, x2, y2 = box.astype(int)
    #             currentArray = np.array([x1, y1, x2, y2, int(p[5])])  # tracking
    #             detections = np.vstack((detections, currentArray))  # tracking
    # tracker_rs2 = Tracker.update(detections)
    for result in rs2:
        boxes = result.boxes.cpu().numpy()
        conf = result.boxes.conf.numpy()
        r = boxes.xyxy
        class_label = boxes.cls
        for box, label,conf in zip(tracker_rs, class_label,conf):
            if label == 0 and conf > 0.5: #label 1 = no helmet, label 0 = helmet, conf = confidence of boxes
                # print(x1)

                x1, y1, x2, y2,id2 = box.astype(int)
                helmet += 1

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    print("Empty.")
                else:
                    # Save the cropped image as JPEG
                    # cv2.imwrite("output_data/helmet{}.jpg".format(helmet), crop)  # save the crop
                    if id2 not in track_list2:
                        track_list2.append(id)
                        new_img = P.save_image(img, id2, x1, y1, x2, y2)  # save image
                    if new_img is not None:
                        img_dect = P.plate_detection(new_img)
                        P.number_plate_extract(img_dect)
                    else:
                        continue



    canvas = img.copy()

    for line in lines:
        cv2.line(canvas, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)

    if drawing:
        cv2.line(canvas, line_start, line_end, (0, 255, 0), 2)
    result = cv2.addWeighted(img, 1, canvas, 0.5, 0)
    stacked = cv2.hconcat([img, result])
    cv2.imshow("Image", stacked)
    k = cv2.waitKey(5)

    if k == ord('c'):
        canvas = img.copy()
        lines = []

    elif k == 27:
        break

cap.release()
cv2.destroyAllWindows()