# -*- coding: utf-8 -*-
import cv2
import numpy as np
from sklearn import *
import dlib
import time

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

detector = dlib.get_frontal_face_detector()
font = cv2.FONT_HERSHEY_SIMPLEX


def linerf(x):
    y = -118 * x + 115
    return y


def loadVIP(path):
    frtrain1 = np.genfromtxt(path, delimiter=' ', dtype=str)
    Xtrain = frtrain1[:, 0:128]
    Ytrain = frtrain1[:, 128:129].ravel()
    return Xtrain, Ytrain


def predictImg(img, clf):
    # 获取所有人脸位置
    t0 = time.time()
    faces = detector(img, 1)
    t00 = time.time()
    if len(faces) > 0:
        for face in faces:
            left = face.left()
            top = face.top()
            right = face.right()
            bottom = face.bottom()
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 1)
            t1 = time.time()
            shape = predictor(img, face)
            t2 = time.time()
            tezhengzhi128 = face_rec_model.compute_face_descriptor(img, shape)  # 128维特征向量
            t3 = time.time()
            tzlist = []
            for i in tezhengzhi128:
                tzlist.append(i)
            resname = clf.predict([tzlist])
            distance, ind = clf.kneighbors([tzlist], n_neighbors=1, return_distance=True)
            score = linerf(distance)
            score = min(98, score)
            if distance[0][0] > 0.4:
                resname[0] = 'unknown'
            t4 = time.time()
            print("获取人脸所用时间%f,获取特征点所用时间%f，转化128特征向量所用时间%f，循环所用时间%f" % (t00 - t0, t2 - t1, t3 - t2, t4 - t3))
            res = resname[0]
            if res != 'unknown':
                res = res + 'score:' + str(score)
            cv2.putText(img, res, (right, bottom), font, 0.5, (255, 255, 255), 2)
    return img


# 加载训练集
Xtrain, Ytrain = loadVIP("F:\\test\\vip\\vip.txt")
loadtime1 = time.time()
clf = neighbors.KNeighborsClassifier(algorithm="ball_tree", metric='euclidean', n_neighbors=1)
clf.fit(Xtrain, Ytrain)
loadtime2 = time.time()
print("建模时间为：", loadtime2 - loadtime1)

cap = cv2.VideoCapture(0)  # 打开1号摄像头
success, frame = cap.read()
while success:
    success, frame = cap.read()  # 读取一桢图像，这个图像用来获取它的大小
    # cv2.imshow("test", frame)  # 显示图像

    t1 = time.time()
    img = predictImg(frame, clf)
    t2 = time.time()
    print('总时间：', t2 - t1)
    cv2.imshow("test", img)  # 显示图像
    key = cv2.waitKey(1)
    key = cv2.waitKey(1)
    c = chr(key & 255)
    if c in ['q', 'Q', chr(27)]:
        break
cv2.destroyAllWindows