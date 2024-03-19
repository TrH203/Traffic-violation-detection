import yolov5
import cv2
import time

import numpy
import pandas
from sort import *
from process import Process
from configparser import ConfigParser
import ultralytics
from ultralytics import YOLO

model = yolov5.load("./weights/vehicle_detect.pt")
cap = cv2.VideoCapture("./source_videos/moto.mp4")

mask = cv2.imread("./source_images/maskmoto.png")

class_names = ['car', 'moto', 'truck', 'bus', 'bycycle']

# Initialize variables for FPS calculation
prev_time = 0
fps = 0

while True:
    success, img = cap.read()
    masked_img = cv2.bitwise_and(img, mask)
    rs = model(img)
    predictions = rs.pred[0]
    for p in predictions:
        score = round(float(p[4]), 2)
        x1, y1, x2, y2 = p[:4]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if True:
            name = class_names[int(p[5])]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(img, f'{name}-{score}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        2)  # draw name and score

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Display FPS on the image
    cv2.putText(img, f'FPS: {round(fps,2)}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)