# -*- coding: utf-8 -*-
import cv2
import math
import numpy as np
import copy
from enum import Enum

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
    _, contours, _ = cv2.findContours(array, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
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
                cv2.circle(array, far, 3, (0,0,255), -1)
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
    width, height = array.shape[0:2]  # 获取图片长宽
    array = cv2.resize(array, (int(height / 3), int(width / 3)))  # 缩放
    event = Event.NONE
    img = copy.deepcopy(array)
    array = _remove_background(array)  # 移除背景, add by wnavy
    thresh = _bodyskin_detetc(array)
    cv2.imshow('thresh', thresh)
    contours = _get_contours(thresh.copy())  # 计算图像的轮廓
    largecont = max(contours, key=lambda contour: cv2.contourArea(contour))
    hull = cv2.convexHull(largecont, returnPoints=False)  # 计算轮廓的凸点
    defects = cv2.convexityDefects(largecont, hull)  # 计算轮廓的凹点
    if defects is not None:
        # 利用凹陷点坐标, 根据余弦定理计算图像中锐角个数
        img, ndefects = _get_defects_count(img, largecont, defects, verbose=verbose)
        cv2.imshow('img', img)
        # 根据锐角个数判断手势, 会有一定的误差
        if ndefects == 0:
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
    cv2.imshow('frame', frame)
    return event

#打印版本
print("OpenCV Version:" + cv2.__version__)
# 读取图片
img = cv2.imread('f1.png', 1)
#grdetect(img,True)
# 1.先找到轮廓
ycrcb = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb) #分解为YUV图像，得到CR分量
(_,cr,_) = cv2.split(ycrcb)  #拆分颜色通道,顺序是b,g,r
cr1 = cv2.GaussianBlur(cr,(5,5),0) #(5,5)表示高斯矩阵的长与宽都是5，标准差取0
_,thresh = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
image, contours, hierarchy = cv2.findContours(thresh, 3, 2)
cnt = contours[0]
# 2.进行多边形逼近，得到多边形的角点
approx = cv2.approxPolyDP(cnt, 3, True)
# 3.画出多边形
image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.polylines(image, [approx], True, (0, 255, 0), 2)
cv2.polylines(image,[cnt],True,(0,0,255),2)
cv2.imshow("image",image)

cv2.waitKey()