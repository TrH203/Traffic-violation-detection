import cv2
import numpy as np
import yolov5
import os
from tool import convert_to_list, rotate_and_crop
# from tensorflow.keras.models import load_model


class Process():
    def __init__(self,save_path) -> None:
        self.save_path = save_path

    def load_number_plate_picture(self, link):
        # load the model which cut only licence plate
        self.model_NPP = yolov5.load(link)
        self.id = 1

    def load_number_plate(self, link):
        #load the model which detect the number in this lincense plate
        self.model_NP = yolov5.load(link)



    def save_image(self,img,id,x1,y1,x2,y2):   
        self.id = id

        # alpha = d[class_name] # Bigger image than obj
        alpha = 50
        if img is not None:
            new_img = img[y1-alpha:y2 + alpha, x1-alpha :x2 + alpha]
            des = os.path.join(self.save_path,f"image{self.id}.jpg")
            cv2.imwrite(des,new_img)
            # self.plate_detection(new_img)
        return new_img
        
    def plate_detection(self,img):
        # img = cv2.resize(img,(224,224))
        rs = self.model_NPP(img) # use model

        predictions = rs.pred[0]  
        # print(predictions)

        for p in predictions:
            # score = round(float(p[4]),2)
            x1,y1,x2,y2 = p[:4]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1,y1,x2,y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),2)

            # cv2.putText(img,f'{name}-{score}',(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            img = img[y1:y2,x1:x2]
        des = os.path.join(self.save_path,f"plate_image{int(self.id)}.jpg")

        cv2.imwrite(des,img)
        return img

    def number_plate_extract(self,img):
        # init class name
        img = rotate_and_crop(img)
        class_names = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','K','L','M','N','P',
                        'S','T','U','V','X','Y','Z','0']
        

        # Resize help model more efficiently
        if img.shape[0] > 100 and img.shape[1] > 100:
            img = cv2.resize(img,(640,640))
        else:
            img = cv2.resize(img,(224,224))


        rs = self.model_NP(img) # use model

        predictions = rs.pred[0] # get predict 


        l = []
        # get boxes, scores, class name
        for p in predictions:
            score = round(float(p[4]),2)
            x1,y1,x2,y2 = p[:4]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1,y1,x2,y2)

            name = class_names[int(p[5])]
            # print(name,score,(x1,y1,x2,y2))
            l.append([name,x1,y1,x2,y2])

            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),2)

            # cv2.line(img,(line[0],line[1]),(line[2],line[3]),color=(255,0,0),thickness=2) # draw the line

            # cv2.putText(img,f'{name}',(x1,max(30,y1)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

        try:
            rs = convert_to_list(l)
            pass
        except:
            rs = "Unknown"
        with open(os.path.join(self.save_path,f"plate_number{self.id}.txt"),"a") as f:
            f.write(str(rs))
