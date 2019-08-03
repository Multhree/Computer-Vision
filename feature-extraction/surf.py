#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/2 下午 04:19
# @Author  : YuXin Chen

import cv2

def surf(filename):
    img = cv2.imread(filename) # 读取文件
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转化为灰度图
    sift = cv2.xfeatures2d_SURF.create()
    keyPoint, descriptor = sift.detectAndCompute(img, None) # 特征提取得到关键点以及对应的描述符（特征向量）
    return img, keyPoint, descriptor

def main():
    img, kp, des = surf('./pic/doraemon1.jpg')
    img = cv2.drawKeypoints(img,kp,None)
    cv2.imshow('surf', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()