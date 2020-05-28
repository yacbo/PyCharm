# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import cv2
import math
import numpy as np
import copy
import dlib
import matplotlib.pyplot as plt
from enum import Enum
import face_recognition
import os
import json

#给人像画矩形框
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

#给人像描绘特征点
def marklabels(img):
    face_landmarks_list = face_recognition.face_landmarks(img)
    if len(face_landmarks_list) != 0:
        for x in range(len(face_landmarks_list)):
            for index, name in enumerate(face_landmarks_list[x]):
                pt = face_landmarks_list[x].get(name)
                for i in range(len(pt)):
                    # pt_pos = (pt[i].x, pt[i].y)
                    cv2.circle(img, pt[i], 2, (0, 0, 255), 2)

#给人像描绘特征点2
def detectmarks(img):
    faces = detector(img, 1)  # 检测到的人像,参数1表示扩大一倍再检测，可以检测更多人像
    if len(faces):
        print('==> Found %d face in this image.' % len(faces))
        for i in range(len(faces)):
            landmarks = np.matrix([[p.x, p.y] for p in predictor(img, faces[i]).parts()])
            for point in landmarks:
                pos = (point[0, 0], point[0, 1])
                cv2.circle(img, pos, 2, color=(0, 255, 0), thickness=1)
    else:
        print('Face not found!')

# opencv读取图片是BRG通道的，需要专成RGB
def showbgr2rgb(img,imgcp):
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


# detector = dlib.cnn_face_detection_model_v1('../mmod_human_face_detector.dat')
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('../dlib_face_recognition_resnet_model_v1.dat')

imagePath = '../Picture/01/db/'  # 图像的目录
data = np.zeros((1, 128))  # 定义一个128维的空向量data
label = []  # 定义空的list存放人脸的标签

for file in os.listdir(imagePath):  # 开始一张一张索引目录中的图像
    if '.jpg' in file or '.png' in file:
        fileName = file
        labelName = file.split('_')[0]  # 获取标签名
        print('current image: ', file)
        print('current label: ', labelName)

        img = cv2.imread(imagePath + file)  # 使用opencv读取图像数据
        if img is None:
            continue
        if img.shape[0] * img.shape[1] > 500000:  # 如果图太大的话需要压缩，这里像素的阈值可以自己设置
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        dets = detector(img, 1)  # 使用检测算子检测人脸，返回的是所有的检测到的人脸区域
        for k, d in enumerate(dets):
            # rec = dlib.rectangle(d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom())
            rec = dlib.rectangle(d.left(), d.top(), d.right(), d.bottom())
            shape = sp(img, rec)  # 获取landmark
            face_descriptor = facerec.compute_face_descriptor(img, shape)  # 使用resNet获取128维的人脸特征向量
            faceArray = np.array(face_descriptor).reshape((1, 128))  # 转换成numpy中的数据结构
            data = np.concatenate((data, faceArray))  # 拼接到事先准备好的data当中去
            label.append(labelName)  # 保存标签
            cv2.rectangle(img, (rec.left(), rec.top()), (rec.right(), rec.bottom()), (0, 255, 0), 2)  # 显示人脸区域
        cv2.waitKey(100)
        cv2.imshow('image', img)

data = data[1:, :]  # 因为data的第一行是空的128维向量，所以实际存储的时候从第二行开始
np.savetxt('01Data.txt', data, fmt='%f')  # 保存人脸特征向量合成的矩阵到本地

labelFile = open('01label.txt', 'w')
json.dump(label, labelFile)  # 使用json保存list到本地
labelFile.close()

cv2.destroyAllWindows()

