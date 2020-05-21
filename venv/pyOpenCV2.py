# -*- coding: utf-8 -*-
import cv2
import numpy as np

# cap = cv2.VideoCapture("rtsp://admin:admin123@10.129.74.198:554/cam/realmonitor?channel=1&subtype=1")
cap = cv2.VideoCapture("rtsp://admin:1234qwer@10.129.74.230:554/live")
while(1):
    ret,frame = cap.read()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    mask = cv2.inRange(hsv,lower_blue,upper_blue)
    res = cv2.bitwise_and(frame,frame,mask=mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5)&0xFF
    if k==27:
        break

cv2.destroyAllWindows()
