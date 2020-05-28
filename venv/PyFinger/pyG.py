# -*- coding: utf-8 -*-
import numpy as np
import cv2
print("OpenCV Version:" + cv2.__version__);

# 读入图像
'''
使用函数 cv2.imread() 读入图像。这幅图像应该在此程序的工作路径，或者给函数提供完整路径.
警告：就算图像的路径是错的，OpenCV 也不会提醒你的，但是当你使用命令print(img)时得到的结果是None。
'''
img = cv2.imread("Picture/0.jpg", cv2.IMREAD_COLOR)
'''
imread函数的第一个参数是要打开的图像的名称(带路径)
第二个参数是告诉函数应该如何读取这幅图片. 其中
	cv2.IMREAD_COLOR 表示读入一副彩色图像, alpha 通道被忽略, 默认值
	cv2.IMREAD_ANYCOLOR 表示读入一副彩色图像
	cv2.IMREAD_GRAYSCALE 表示读入一副灰度图像
	cv2.IMREAD_UNCHANGED 表示读入一幅图像，并且包括图像的 alpha 通道
'''
# 显示图像
'''
使用函数 cv2.imshow() 显示图像。窗口会自动调整为图像大小。第一个参数是窗口的名字，
其次才是我们的图像。你可以创建多个窗口，只要你喜欢，但是必须给他们不同的名字.
'''
cv2.imshow("image", img) # "image" 参数为图像显示窗口的标题, img是待显示的图像数据
cv2.waitKey(0) #等待键盘输入,参数表示等待时间,单位毫秒.0表示无限期等待
cv2.destroyAllWindows() # 销毁所有cv创建的窗口
# 也可以销毁指定窗口:
#cv2.destroyWindow("image") # 删除窗口标题为"image"的窗口
# 保存图像
'''
使用函数 cv2.imwrite() 来保存一个图像。首先需要一个文件名，之后才是你要保存的图像。
保存的图片的格式由后缀名决定.
'''
#cv2.imwrite(imname + "01.png", img)
cv2.imwrite("Picture/01.jpg", img)

"""读取视频并检测人像"""
cap = cv2.VideoCapture("rtsp://admin:1234qwer@10.129.74.230:554/live")
while True:
    ret, img = cap.read()
    if ret:
        width, height = img.shape[0:2]
        img = cv2.resize(img, (int(height / 3), int(width / 3)))
        fgbg = cv2.createBackgroundSubtractorMOG2()  # 利用BackgroundSubtractorMOG2算法消除背景
        fgmask = fgbg.apply(img)
        kernel = np.ones((5, 5), np.uint8)
        fgmask = cv2.erode(fgmask, kernel, iterations=1)  # 膨胀
        res = cv2.bitwise_and(img, img, mask=fgmask)
        ycrcb = cv2.cvtColor(res, cv2.COLOR_BGR2YCrCb) # 分解为YUV图像,得到CR分量
        (_, cr, _) = cv2.split(ycrcb)
        cr1 = cv2.GaussianBlur(cr, (5, 5), 0)  # 高斯滤波
        _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # OTSU图像二值化

    img, contours, hierarchy = cv2.findContours(skin, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours): # 获取轮廓
        cv2.drawContours(img[0:350, 380:700], contours, i, (255, 0, 0), 1) # 绘制轮廓
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img[0:350, 380:700], (x, y), (x + w, y + h), (100, 100, 0), 1)

    cv2.imshow('frame', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break