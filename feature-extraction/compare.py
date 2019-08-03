#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/2 下午 04:31
# @Author  : YuXin Chen

import cv2
import numpy as np
from sift import *
from surf import *
from orb import *

def compare(filename):
    imgs = []
    keyPoint = []
    descriptor = []
    img, keyPoint_temp, descriptor_temp = sift(filename)
    keyPoint.append(keyPoint_temp)
    descriptor.append(descriptor_temp)
    imgs.append(img)
    img, keyPoint_temp, descriptor_temp = surf(filename)
    keyPoint.append(keyPoint_temp)
    descriptor.append(descriptor_temp)
    imgs.append(img)
    img, keyPoint_temp, descriptor_temp = orb(filename)
    keyPoint.append(keyPoint_temp)
    descriptor.append(descriptor_temp)
    imgs.append(img)
    return imgs, keyPoint, descriptor

def match(filename1, filename2, method):
    if(method == 'sift'):
        img1, kp1, des1 = sift(filename1)
        img2, kp2, des2 = sift(filename2)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)  # sift的normType应该使用NORM_L2或者NORM_L1
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        knnMatches = bf.knnMatch(des1, des2, k=1)  # drawMatchesKnn
    if (method == 'surf'):
        img1, kp1, des1 = surf(filename1)
        img2, kp2, des2 = surf(filename2)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)  # surf的normType应该使用NORM_L2或者NORM_L1
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        knnMatches = bf.knnMatch(des1, des2, k=1)  # drawMatchesKnn
    if(method == 'orb'):
        img1, kp1, des1 = orb(filename1)
        img2, kp2, des2 = orb(filename2)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) # orb的normType应该使用NORM_HAMMING
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        knnMatches = bf.knnMatch(des1, des2, k = 1) # drawMatchesKnn
    # 过滤
    for m in matches:
        for n in matches:
            if(m.distance >= n.distance*0.75):
                matches.remove(m)
                break
    print('%s size of kp: %d, after filtering: %d' % (method, len(des1),len(matches)))
    img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], img2, flags=2)
    cv2.imshow(method, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def main():
    method = ['sift','surf','orb']
    # imgs, kp, des = compare('./pic/doraemon1.jpg')
    # for i in range(3):
    #     img = cv2.drawKeypoints(imgs[i], kp[i], None)
    #     cv2.imshow(method[i], img)
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()
    # print("sift len of des: %d, size of des: %d" % (len(des[0]), len(des[0][0])))
    # print("surf len of des: %d, size of des: %d" % (len(des[1]), len(des[1][0])))
    # print("orb len of des: %d, size of des: %d" % (len(des[2]), len(des[2][0])))
    for i in range(3):
        match('./pic/wechat1.jpg', './pic/wechat2.png', method[i])
if __name__ == '__main__':
    main()