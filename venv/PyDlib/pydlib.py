# coding=utf-8
import cv2
import dlib

detector = dlib.get_frontal_face_detector()  #使用默认的人类识别器模型


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

#cap = cv2.VideoCapture("rtsp://admin:1234qwer@10.129.74.230:554/live")
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("../Video/2.flv")
while (1):
    ret, img = cap.read()
    if img is None:
        continue
    width, height = img.shape[0:2]  # 获取图片长宽
    img = cv2.resize(img, (int(height / 2), int(width / 2)))  # 缩放
    # cv2.imshow("img", img)
    discern(img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()