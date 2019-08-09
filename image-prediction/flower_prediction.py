#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/9 下午 08:55
# @Author  : YuXin Chen

from imageai.Prediction.Custom import CustomImagePrediction
import os

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath("./model/model_ex-075_acc-0.866071.h5")
prediction.setJsonPath("./model/model_class.json")
prediction.loadModel(num_objects=5)

predictions, probabilities = prediction.predictImage("./pic/test/tulips.jpg", result_count=5)

for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction + " : " + eachProbability)