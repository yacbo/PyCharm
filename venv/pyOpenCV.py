# -*- coding: utf-8 -*-
import cv2
import numpy as np

print("OpenCV Version:" + cv2.__version__);

# img = cv2.imread("./0.jpg")
# cv2.namedWindow("Image")
# cv2.imshow("pic",img)
# cv2.waitKey(1000)


# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("pic",gray)
# cv2.waitKey(1000)
# cv2.imwrite("pic.jpg",gray)


"""读取视频并检测人像"""
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#cap = cv2.VideoCapture("rtsp://admin:admin123@10.129.74.198:554/cam/realmonitor?channel=1&subtype=1")
cap = cv2.VideoCapture("rtsp://admin:1234qwer@10.129.74.230:554/live")

while True:
    ret, img = cap.read()
    if ret:
        width, height = img.shape[0:2]
        img = cv2.resize(img, (int(height / 3), int(width / 3)))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 3)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()