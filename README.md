# Traffic-violation-detection
Scientific research project at Duy Tan University (DTU) (INPROCESS)

## Overview
This project focuses on detecting traffic violations using YOLOv5 models in conjunction with OpenCV (cv2) for implementation.

<img width="960" alt="Screenshot 2023-11-19 at 14 50 53" src="https://github.com/TrH203/Traffic-violation-detection/assets/96675314/b945dfbd-220b-40c4-a162-bb4f988d5d5e">

Project Structure
### 1. Models Used
I employ three YOLOv5 models for different aspects of traffic violation detection:

- [Vehicle detection]: Use to detect Vehicle
- [Plate detection]: From Vehicle then extract plate 
- [Number Plate detection]: From plate then extract number in plate
### 2. How it works ?

We implement **yolov5** to train all of these 3 model to solve each problem.
#### 1. Pre-processing:
Because in the reality, the image that I capture is not in high quality, but the train images is very clear. So i decided to make blur image to enhance performance of _Number Plate Detection_ model.

- Before
  
<img width="300" alt="Pre-processing" src="https://github.com/TrH203/Traffic-violation-detection/assets/96675314/b8a34290-6ce4-4fac-8c98-69d0a2c983f5">

- After

<img width="300" alt="Pre-processing2" src="https://github.com/TrH203/Traffic-violation-detection/assets/96675314/bf95f393-e9e2-49b7-9c32-c2b7a0d2adce">

#### 2. Train-model:
I use kangle for train data with yolov5. see more at ==> https://www.kaggle.com/hienhoangtrong/code

##### Read more about YOLOV5  ==> `https://github.com/ultralytics/yolov5`


#### 3. Result:
I create a pipeline between 3 model that the output of this model is the output of that model.
- First model (Vihecle detection)
  Input: Images, Mask, Separation Line ( for Lane encroachment violation )
  Output:
    The image contain the violation object
  example:
  
  ![image6](https://github.com/TrH203/Traffic-violation-detection/assets/96675314/596a63a0-4342-4096-83f3-53435c232839)

  

- Second model (Plate detection)
  Output:
    The image contain the violation object's plate
  
  <img width="200" alt="Plate" src="https://github.com/TrH203/Traffic-violation-detection/assets/96675314/22022b1a-5a5f-4bff-8643-f8ee3f1c2399">

- Last model (Number plate detection)
  Output:
    The number in plate then I extract the number into text
  example:
  
  `59N12345, 76E152202,....`



### 3. Implementation

Step 1:
- Install all lib, framework.
- Modify the input path, output path `project.ini`.
- Setting mask, graphic setting in `project.ini`.

Step 2:
- Run file
   ```
   python.3.7 yolo-detect.py
   ```
- If you have GPU, setting to run that, see in 

## How to Run
Navigate to the project directory:
```
python3.7 yolo-detect.py
```


### Data
View my data in Kaggle now : https://www.kaggle.com/hienhoangtrong/datasets


### Weight
helmet: https://drive.google.com/file/d/1JI9Gk-mjYQEf9K27XGWDeeyWR81TAj6c/view?usp=sharing
