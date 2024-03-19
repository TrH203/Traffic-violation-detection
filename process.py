import cv2
import numpy as np
import yolov5
import os
from tool import convert_to_list, rotate_and_crop
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# from tensorflow.keras.models import load_model
from workWithDatabase import DatabaseConnector;
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



    def save_image(self,img,id,x1,y1,x2,y2, image_name=None):
        self.id = id

        # alpha = d[class_name] # Bigger image than obj
        alpha = 50
        if img is not None:
            if image_name == "image":
                new_img = img[y1 - alpha:y2 + alpha, x1 - alpha:x2 + alpha]
                des = os.path.join(self.save_path,f"image{self.id}.jpg")
            else:
                new_img = img[y1:y2*3, x1:x2]
                des = os.path.join(self.save_path, f"helmet{self.id}.jpg")
            cv2.imwrite(des,new_img)
            # self.plate_detection(new_img)
        return new_img
        
    def plate_detection(self,img):
        # img = cv2.resize(img,(224,224))
        rs = self.model_NPP(img) # use model
        if img is None:
            print(f"Error: Could not load image '{img}'")
            return None
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
        if img.size == 0:
            print("Error: The cropped image is empty. No license plates detected.")
            return None
        des = os.path.join(self.save_path,f"plate_image{int(self.id)}.jpg")

        cv2.imwrite(des,img)
        return img

    def number_plate_extract(self, img):
        try:
            # init class name
            if img is None or img.size == 0:
                print("Error: The cropped image is empty. No license plates detected.")
                return None

            img = rotate_and_crop(img)
            class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K',
                            'L', 'M', 'N', 'P',
                            'S', 'T', 'U', 'V', 'X', 'Y', 'Z', '0']

            # Resize help model more efficiently
            if img.shape[0] > 100 and img.shape[1] > 100:
                img = cv2.resize(img, (640, 640))
            else:
                img = cv2.resize(img, (224, 224))

            rs = self.model_NP(img)  # use model
            predictions = rs.pred[0]  # get predict

            l = []
            # get boxes, scores, class name
            for p in predictions:
                score = round(float(p[4]), 2)
                x1, y1, x2, y2 = p[:4]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                name = class_names[int(p[5])]
                l.append([name, x1, y1, x2, y2])

            rs = convert_to_list(l)
            pass
        except Exception as e:
            print(f"Error occurred: {e}")
            rs = "Unknown"

        with open(os.path.join(self.save_path, f"plate_number{self.id}.txt"), "a") as f:
            f.write(str(rs))
            
