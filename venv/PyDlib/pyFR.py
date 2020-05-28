# -*- coding: utf-8 -*-
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

#根据128维特征向量和label值 找出最相似的
def findNearestClassForImage(face_descriptor, faceLabel):
    temp =  face_descriptor - data
    e = np.linalg.norm(temp,axis=1,keepdims=True)
    min_distance = e.min()
    # print('distance: ', min_distance)
    similar = 1- float('%.4f' %min_distance)
    print('similar: ',str(similar*100) + '%')
    if min_distance > threshold:
        return 'other'
    index = np.argmin(e)
    return faceLabel[index],similar    #返回label中对应的图片名


def recognition(img):
    height,width = img.shape[0:2]
    dets = detector(img, 1)   #获取到人像
    frame = copy.deepcopy(img)

    for d in dets:
    # for k, d in enumerate(dets):   #遍历人像
        # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
        # rec = dlib.rectangle(d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom())
        rec = dlib.rectangle(d.left(), d.top(), d.right(), d.bottom())
        # print(rec.left(), rec.top(), rec.right(), rec.bottom())
        # cut = img[rec.top():rec.bottom(),d.left():rec.right()]     #高区间,宽区间
        shape = predictor(img, rec)
        tezhengzhi128 = facerec.compute_face_descriptor(img, shape)  # 128维特征向量

        class_pre,similar = findNearestClassForImage(tezhengzhi128, label)
        print(class_pre) #打印从lable中找到的人像类
        cv2.rectangle(img, (rec.left(), rec.top() + 10), (rec.right(), rec.bottom()), (0, 255, 0), 2)
        cv2.putText(img, class_pre +'  '+str(100*similar)+'%', (rec.left(), rec.top()), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        frame = cv2.imread('../Picture/test/' + class_pre)  # 使用opencv读取图像数据
    # cv2.imshow('image', img)
    # cv2.imshow('frame', frame)
    return img,frame,similar


# detector = dlib.cnn_face_detection_model_v1('../mmod_human_face_detector.dat')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('../dlib_face_recognition_resnet_model_v1.dat')
threshold = 0.54

labelFile = open('label.txt', 'r')
label = json.load(labelFile)  # 载入本地人脸库的标签
labelFile.close()
data = np.loadtxt('faceData.txt', dtype=float)  # 载入本地人脸特征向量

imagePath = '../Picture/test1/'  # 图像的目录

for file in os.listdir(imagePath):  # 开始一张一张索引目录中的图像
    if '.jpg' in file or '.png' in file:
        img = cv2.imread(imagePath + file)  # 使用opencv读取图像数据
        if img is None:
            continue
        if img.shape[0] * img.shape[1] > 500000:  # 如果图太大的话需要压缩，这里像素的阈值可以自己设置
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        img,frame,similar = recognition(img)
        if similar > 0.62:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(10, 6))
            plt.subplot(121)
            plt.imshow(img)
            plt.axis('off')

            plt.subplot(122)
            plt.imshow(frame)
            plt.axis('off')
            plt.show()
            cv2.waitKey(2500)
# while (1):
#     ret, frame = cap.read()
#     # frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
#     recognition(frame)
#     # videoWriter.write(frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
cv2.destroyAllWindows()