# -*- coding: utf-8 -*-
import cv2
import math
import numpy as np
import copy
import dlib
import matplotlib.pyplot as plt
from enum import Enum


def discern(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)
    for face in dets:
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 1)
    cv2.imshow("image", img)

#打印版本
print("OpenCV Version:" + cv2.__version__)
# 读取图片
img = cv2.imread('../Picture/f1.png', 1)
predictor_path = '../shape_predictor_68_face_landmarks.dat'

img = cv2.imread('../Picture/aa.png')
imgcp = copy.deepcopy(img)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

faces = detector(img, 1)   #检测到的人像,参数1表示扩大一倍再检测，可以检测更多人像
if len(faces):
    print('==> Found %d face in this image.' % len(faces))
    for i in range(len(faces)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img, faces[i]).parts()])
        for point in landmarks:
            pos = (point[0, 0], point[0, 1])
            cv2.circle(img, pos, 2, color=(0, 255, 0),thickness=1)
else:
    print ('Face not found!')

# opencv读取图片是BRG通道的，需要专成RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgcp = cv2.cvtColor(imgcp, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 6))
plt.subplot(121)
plt.imshow(imgcp)
plt.axis('off')

plt.subplot(122)
plt.imshow(img)
plt.axis('off')
plt.show()
cv2.waitKey()