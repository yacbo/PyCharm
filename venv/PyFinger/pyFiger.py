# -*- coding: utf-8 -*-
#import cv2
#img = cv2.imread("aaa.jpg")
# print(img);
# 在窗口中显示图像
#cv2.imshow("Image", img)
# 如果不添waitKey ，在IDLE中执行窗口直接无响应。在命令行中执行的话，则是一闪而过
#cv2.waitKey(10000)
#cv2.destroyAllWindows()

import cv2
import math
import numpy as np
import copy
from enum import Enum
import dlib

class Event(Enum):
    NONE=0
    ZERO=1
    TWO=2
    THREE=3
    FOUR=4
    FIVE=5

#移除视频数据的背景噪声
def _remove_background(frame):
    fgbg = cv2.createBackgroundSubtractorMOG2()
    fgmask = fgbg.apply(frame)
    kernel = np.ones((3,3),np.uint8)
    fgmask =cv2.erode(fgmask,kernel,iterations=1)
    res = cv2.bitwise_and(frame,frame,mask=fgmask)
    return res

#检查肤色
def _bodyskin_detetc(frame):
    ycrcb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb) #分解为YUV图像，得到CR分量
    (_,cr,_) = cv2.split(ycrcb)  #拆分颜色通道,顺序是b,g,r
    cr1 = cv2.GaussianBlur(cr,(5,5),0) #(5,5)表示高斯矩阵的长与宽都是5，标准差取0
    _,skin = cv2.threshold(cr1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return skin

# 检测图像中的凸点(手指)个数
def _get_contours(array):
    # 利用findContours检测图像中的轮廓, 其中返回值contours包含了图像中所有轮廓的坐标点
    img, contours, _ = cv2.findContours(array, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours

def _get_eucledian_distance(start,end):
    value = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    return value

# 根据图像中凹凸点中的 (开始点, 结束点, 远点)的坐标, 利用余弦定理计算两根手指之间的夹角, 其必为锐角, 根据锐角的个数判别手势.
def _get_defects_count(array, contour, defects, verbose=False):
    ndefects = 0
    for i in range(defects.shape[0]):
        s, e, f, _ = defects[i, 0]
        beg = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        a = _get_eucledian_distance(beg, end)
        b = _get_eucledian_distance(beg, far)
        c = _get_eucledian_distance(end, far)
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # * 57
        if angle <= math.pi / 2:  # 90:
            ndefects = ndefects + 1
            if verbose:
                cv2.circle(array, far, 3, (0,0,255), -1)     #画实心圆点
        if verbose:
            cv2.line(array, beg, end, (0,0,255), 1)
    return array, ndefects

#显示手势
def _show_figer_num(ndefects,img):
    frame = copy.deepcopy(img)
    figernum = ndefects + 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, str(figernum), (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    return figernum,frame

#方法1检测手势
def grdetect(array, verbose=False):
    if array is None :
        print("array is None")
        return
    # width, height = array.shape[0:2]  # 获取图片长宽
    # array = cv2.resize(array, (int(height / 3), int(width / 3)))  # 缩放
    img = copy.deepcopy(array)
    event = Event.NONE
    array = _remove_background(array)  # 移除背景, add by wnavy
    # cv2.imshow('array',array)
    thresh = _bodyskin_detetc(array)
    cv2.imshow('thresh', thresh)   #########################################
    contours = _get_contours(thresh.copy())  # 计算图像的轮廓
    largecont = max(contours, key=lambda contour: cv2.contourArea(contour))
    hull = cv2.convexHull(largecont, returnPoints=False)  # 计算轮廓的凸点
    defects = cv2.convexityDefects(largecont, hull)  # 计算轮廓的凹点

    areacntlargecont = 1.0*cv2.contourArea(largecont) # 计算轮廓面积
    # print('largecont:'+str(areacntlargecont))

     #外接圆
    (x, y), radius = cv2.minEnclosingCircle(largecont)
    center = (int(x), int(y))
    radius = int(radius)
    array = cv2.circle(array, center, radius, (0, 255, 0), 1)
    cv2.imshow("array",array)
    areacntradius = 1.0*math.pi *radius*radius # 计算轮廓面积

    #外接矩形
    # x, y, w, h = cv2.boundingRect(largecont)
    # array = cv2.rectangle(array, (x, y), (x + w, y + h), (0, 255, 0), 1)
    # cv2.imshow("array", array)
    # areacntradius = 1.0*w*h  # 计算轮廓面积
    # print('radius:'+str(areacntradius))

    rat = 1.0*areacntlargecont/areacntradius*1.0

    # epsilon = 0.00005 * cv2.arcLength(largecont, True)   #计算轮廓周长
    # approx = cv2.approxPolyDP(largecont, 1, True)  # 进行多边形逼近,得到多边形的角点
    # cv2.polylines(array, [approx], True, (0, 0, 255), 1)
    # cv2.imshow("array", array)
    # hull = cv2.convexHull(approx)  # 计算轮廓的凸点
    # areahull = cv2.contourArea(hull)  # 计算凸点面积
    # areacnt = cv2.contourArea(approx)  # 计算轮廓面积
    # arearatio = ((areahull - areacnt) / areacnt) * 100
    # hull = cv2.convexHull(approx, returnPoints=False)
    # defects = cv2.convexityDefects(approx, hull)

    if defects is not None:
        # 利用凹陷点坐标,根据余弦定理计算图像中锐角个数
        img, ndefects = _get_defects_count(img, largecont, defects, verbose=verbose)
        cv2.imshow('img', img)    ############################################
        # 根据锐角个数判断手势,会有一定的误差
        if ndefects == 0:
             if rat > 0.506:
                  event = Event.NONE
             elif rat < 0.3:
                 event = Event.ZERO
        elif ndefects == 1:
            event = Event.TWO
        elif ndefects == 2:
            event = Event.THREE
        elif ndefects == 3:
            event = Event.FOUR
        elif ndefects == 4:
            event = Event.FIVE
    frame = copy.deepcopy(array)
    cv2.putText(frame, str(event.value), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
    print(str(event.value)+'  :  ' +str(rat))
    cv2.imshow('frame', frame)     ################################################
    return event

print("OpenCV Version:" + cv2.__version__);
#cap = cv2.VideoCapture("rtsp://admin:abcd1234@192.168.1.64:554//Streaming/Channels/1")  # 打开指定路径上的视频文件
#cap=cv2.VideoCapture(0) #打开设备索引号对于设备的摄像头，一般电脑的默认索引号为0
# cap = cv2.VideoCapture("rtsp://admin:1234qwer@10.129.74.230:554/live")   #E:\PycharmProject\venv
cap = cv2.VideoCapture("Video/2.mp4")
while (True):
     ret, frame = cap.read()
     if frame is None:
         continue
     height,width = frame.shape[0:2]   #高-宽
     frame = cv2.resize(frame, (int(width / 3), int(height / 3)))  #缩放 宽-高
     # frame = cv2.flip(frame, 1)  #视频流需要翻转,图片不用翻转
     kernelA = np.ones((3, 3), np.uint8)
     height, width = frame.shape[0:2]  # 高-宽
     # roi = frame[250:400, 210:390]    #高-宽区间
     # cv2.rectangle(frame, (210, 250), (390, 400), (0, 255, 255), 0)   #对角顶点
     cv2.imshow("frame1", frame)
     grdetect(frame,True)
     #在播放每一帧时，使用cv2.waitKey()设置适当的持续时间。如果设置的太低视频就会播放的非常快，如果设置的太高就会播放的很慢。通常情况下25ms就ok
     k = cv2.waitKey(25) & 0xFF
     if k==27:
         break
cap.release()
cv2.destroyAllWindows()

# hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # 对roi区域进行rgb转hsv
# lower_skin = np.array([0, 20, 70], dtype=np.uint8)
# upper_skin = np.array([20, 255, 255], dtype=np.uint8)
# mask = cv2.inRange(hsv, lower_skin, upper_skin)  # 去除背景噪声
# mask = cv2.dilate(mask, kernelA, iterations=4)  # 膨胀 src表示输入的图片， kernel表示方框的大小， iteration表示迭代的次数
# mask = cv2.GaussianBlur(mask, (5, 5), 100)
#
# fgbg = cv2.createBackgroundSubtractorMOG2()  # 利用BackgroundSubtractorMOG2算法消除背景
# fgmask = fgbg.apply(frame)
# kernel = np.ones((3, 3), np.uint8)
# fgmask = cv2.erode(fgmask, kernel, iterations=1)  # 腐蚀
# res = cv2.bitwise_and(frame, frame, mask=fgmask)
# ycrcb = cv2.cvtColor(res, cv2.COLOR_BGR2YCrCb)  # 分解为YUV图像,得到CR分量
# (_, cr, _) = cv2.split(ycrcb)
# cr1 = cv2.GaussianBlur(cr, (5, 5), 0)  # 高斯滤波
# _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # OTSU图像二值化
# cv2.imshow("skin", skin)
#
# img, contours, hierarchy = cv2.findContours(skin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cnt = max(contours, key=lambda x: cv2.contourArea(x))  # 找出面积最大的轮廓
# epsilon = 0.0005 * cv2.arcLength(cnt, True)
# approx = cv2.approxPolyDP(cnt, epsilon, True)  # 进行多边形逼近,得到多边形的角点
# hull = cv2.convexHull(cnt)  # 计算轮廓的凸点
#
# areahull = cv2.contourArea(hull)  # 计算凸点面积
# areacnt = cv2.contourArea(cnt)  # 计算轮廓面积
# arearatio = ((areahull - areacnt) / areacnt) * 100
#
# hull = cv2.convexHull(approx, returnPoints=False)
# defects = cv2.convexityDefects(approx, hull)
# if defects is None:
#     continue
# I = 0
# for i in range(defects.shape[0]):
#     s, e, f, d = defects[i, 0]
#     start = tuple(approx[s][0])
#     end = tuple(approx[e][0])
#     far = tuple(approx[f][0])
#     pt = (100, 180)
#
#     a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
#     b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
#     c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
#     s = (a + b + c) / 2
#     ar = math.sqrt(s * (s - a) * (s - b) * (s - c))
#
#     d = (2 * ar) / a
#     angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
#
#     if angle <= 90 and d > 30:
#         I += 1
#         cv2.circle(roi, far, 3, [255, 0, 0], -1)
#     cv2.line(roi, start, end, [0, 255, 0], 2)
#
# I += 1
# font = cv2.FONT_HERSHEY_SIMPLEX
# if I == 1:
#     if areacnt < 2000:
#         cv2.putText(frame, 'Put hand in the box', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
#     else:
#         if arearatio < 12:
#             cv2.putText(frame, '0', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
#         elif arearatio < 17.5:
#             cv2.putText(frame, 'Best of luck', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
#         else:
#             cv2.putText(frame, '1', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
# elif I == 2:
#     cv2.putText(frame, '2', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
# elif I == 3:
#     if arearatio < 27:
#         cv2.putText(frame, '3', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
#     else:
#         cv2.putText(frame, 'ok', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
# elif I == 4:
#     cv2.putText(frame, '4', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
# elif I == 5:
#     cv2.putText(frame, '5', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
# elif I == 6:
#     cv2.putText(frame, 'reposition', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
# else:
#     cv2.putText(frame, 'reposition', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
#
# cv2.imshow("video", frame)
# cv2.imshow("mask", mask)