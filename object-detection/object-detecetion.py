#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/8 下午 01:26
# @Author  : YuXin Chen

from imageai.Detection import ObjectDetection
import os
import numpy as np

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("./model/yolo.h5")
detector.loadModel()

path = os.getcwd()
input_image_list = os.listdir(path+"\pic\input")
input_image_list = sorted(input_image_list, key = lambda i:len(i),reverse = False)
size = len(input_image_list)
for i in range(size):
    input_image_path = os.path.join(path+"\pic\input", input_image_list[i])
    output_image_path = os.path.join(path+"\pic\output", input_image_list[i])
    detections, extract_detected_objects = detector.detectObjectsFromImage(input_image=input_image_path,
                                                 output_image_path=output_image_path,
                                                 extract_detected_objects=True)
    print('------------------- %d -------------------' % int(i + 1))
    for eachObject in detections:
        print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])
    print('------------------- %d -------------------' % int(i + 1))