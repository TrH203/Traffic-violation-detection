# Traffic Violation Detection

[![Development Status](https://img.shields.io/badge/status-in%20development-orange)]() 
[![Version](https://img.shields.io/badge/version-0.1.0-blue)]() 
[![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)](https://github.com/TrH203/Vietnamese-Speech-to-Sign-Language/network/dependencies) 
[![Contributors](https://img.shields.io/github/contributors/TrH203/Traffic-violation-detection)](https://github.com/TrH203/Vietnamese-Speech-to-Sign-Language/graphs/contributors) 
![Python Version](https://img.shields.io/badge/python-3.10.12-blue)

## Overview

This scientific research project, conducted at Duy Tan University (DTU), aims to develop a robust system for detecting **traffic violations** using **YOLOv5** models in conjunction with **OpenCV**.

<p align="center">
  <img width="960" alt="Project Screenshot" src="https://github.com/TrH203/Traffic-violation-detection/assets/96675314/b945dfbd-220b-40c4-a162-bb4f988d5d5e">
</p>

### Youtube Demo
Link: https://youtu.be/ddhAg65acd0

## Project Structure

### Models Used
The project leverages three distinct YOLOv5 models to address different components of traffic violation detection:

- **Vehicle Detection**: Identifies vehicles on the road.
- **Helmet Detection**: Identifies helmet on the road.
- **Plate Detection**: Extracts the license plate from the detected vehicle.
- **Number Plate Detection**: Extracts the numeric characters from the license plate.

### How It Works

The detection pipeline involves the following steps:

1. **Vehicle Detection**: Locating the vehicle on the road.
2. **Helmet Detection**: (If applicable) Identifying whether the rider is wearing a helmet.
3. **License Plate Detection**: Isolating the vehicle’s license plate.
4. **Number Extraction**: Extracting numeric values from the license plate.

### Pre-processing

Given that real-world images often lack the clarity of training images, pre-processing is crucial. We apply a blurring technique to improve the performance of the Number Plate Detection model.

- **Before Pre-processing**:
  <p align="center">
    <img width="300" alt="Pre-processing" src="https://github.com/TrH203/Traffic-violation-detection/assets/96675314/b8a34290-6ce4-4fac-8c98-69d0a2c983f5">
  </p>

- **After Pre-processing**:
  <p align="center">
    <img width="300" alt="Pre-processing2" src="https://github.com/TrH203/Traffic-violation-detection/assets/96675314/bf95f393-e9e2-49b7-9c32-c2b7a0d2adce">
  </p>

### Training

Model training is performed on Kaggle using YOLOv5. More details can be found in the [Kaggle Notebook](https://www.kaggle.com/hienhoangtrong/code).

For a deeper dive into YOLOv5, refer to the [official documentation](https://github.com/ultralytics/yolov5).

### Results

A pipeline integrates the three models, where each model's output serves as the input for the next.

1. **Vehicle Detection**:
   - **Input**: Images, Mask, Separation Line (for Lane Encroachment Detection).
   - **Output**: Images highlighting the detected violations.
   <p align="center">
     <img width="600" alt="Vehicle Detection" src="https://github.com/TrH203/Traffic-violation-detection/assets/96675314/596a63a0-4342-4096-83f3-53435c232839">
   </p>

2. **Plate Detection**:
   - **Output**: Cropped images containing the vehicle's license plate.
   <p align="center">
     <img width="200" alt="Plate Detection" src="https://github.com/TrH203/Traffic-violation-detection/assets/96675314/22022b1a-5a5f-4bff-8643-f8ee3f1c2399">
   </p>

3. **Number Plate Detection**:
   - **Output**: Extracted numeric values, e.g., `59N12345, 76E152202, ...`

### Data

You can view and explore the dataset used in this project on [Kaggle](https://www.kaggle.com/hienhoangtrong/datasets).

### Pre-trained Weights

Pre-trained weights for the helmet detection model can be downloaded [here](https://drive.google.com/file/d/1JI9Gk-mjYQEf9K27XGWDeeyWR81TAj6c/view?usp=sharing).
