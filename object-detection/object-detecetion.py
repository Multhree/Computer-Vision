#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/8 下午 01:26
# @Author  : YuXin Chen

from imageai.Detection import ObjectDetection
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

def fileStream(detector):
    detections = []
    extract_detected_objects = []
    path = os.getcwd()
    input_image_list = os.listdir(path + "\pic\input")
    input_image_list = sorted(input_image_list, key=lambda i: len(i), reverse=False)
    size = len(input_image_list)
    for i in range(size):
        input_image_path = os.path.join(path + "\pic\input", input_image_list[i])
        output_image_path = os.path.join(path + "\pic\output", input_image_list[i])
        det, obj = detector.detectObjectsFromImage(input_image=input_image_path,
                                                   output_image_path=output_image_path,
                                                   extract_detected_objects=True)
        detections.append(det)
        extract_detected_objects.append(obj)
    return detections, extract_detected_objects

def fileArray(detector):
    detections = []
    extract_detected_objects = []
    path = os.getcwd()
    input_image_list = os.listdir(path + "\pic\input")
    input_image_list = sorted(input_image_list, key=lambda i: len(i), reverse=False)
    size = len(input_image_list)
    for i in range(size):
        input_image_path = os.path.join(path + "\pic\input", input_image_list[i])
        output_image_path = os.path.join(path + "\pic\output", input_image_list[i])
        imgArray = plt.imread(input_image_path)
        det, obj = detector.detectObjectsFromImage(input_image=imgArray,
                                                   input_type='array',
                                                   output_type='array')
        detections.append(det)
        extract_detected_objects.append(obj)
    return detections, extract_detected_objects

def main():
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath("./model/yolo.h5")
    detector.loadModel("fast")
    detections, extract_detected_objects = fileStream(detector)
    # print(detections)
    size = len(detections)
    for i in range(size):
        print('------------------- %d -------------------' % int(i + 1))
        for eachObject in detections[i]:
            print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])
        print('------------------- %d -------------------' % int(i + 1))
    # for i in range(size):
    #     plt.imshow(detections[i])
    #     plt.show()

if __name__ == "__main__":
    main()