#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/9 下午 08:12
# @Author  : YuXin Chen

from imageai.Prediction.Custom import ModelTraining
model_trainer = ModelTraining()
model_trainer.setModelTypeAsResNet()
model_trainer.setDataDirectory("./pic/flower")
model_trainer.trainModel(num_objects=5, num_experiments=100, enhance_data=True, batch_size=48, show_network_summary=True)