import os
import torch
import torch.nn as nn
import numpy as np
import cv2

"""
    什么是阈值？
        阈值是 OpenCV 中的一种技术，它是相对于所提供的阈值分配像素值。在阈值处理中，将每个像素值与阈值进行比较。
    如果像素值小于阈值，则设置为0，否则设置为最大值（一般为255）。阈值是一种非常流行的分割技术，用于将被视为前景的对象与其背景分离。
    阈值是在其任一侧具有两个区域的值，即低于阈值或高于阈值。 在计算机视觉中，这种阈值技术是在灰度图像上完成的。
"""
root = "/home/sci/zyn/test/"
save_path = "/home/sci/zyn/test_res"
imgs = os.listdir(root)
imgs_path = [os.path.join(root, p) for p in imgs]


def getSkewAngle(cvImage):
    """
        得到图片的偏移角度
    :param cvImage: np格式的图片对象
    :return: 角度值
    """
    # 预处理图片
    newImage = cvImage.copy()  # 复制一份
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)  # 转为灰度
    blur = cv2.GaussianBlur(gray, (9, 9), 0)  # 加一点高斯模糊减少噪点
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  # 阈值化。
    cv2.imshow("gray", gray)
    cv2.imshow("blur", blur)
    cv2.imshow("thresh", thresh)

    # 获得矩形结构
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=5)   # 扩张
    cv2.imshow("dilate", dilate)

    # 寻找轮廓
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # print(contours)
    # print(hierarchy)

    # 在最小面积框中查找最大轮廓和环绕
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)

    # Determine the angle. Convert it to the value that was originally used to obtainskewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle


def rotateImage(cvImage, angle: float, filename:str):
    """
        根据角度旋转图像
    :param cvImage:
    :param angle:
    :return:
    """
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.imshow("newImage", newImage)
    cv2.imwrite(filename.split(".")[0]+"_res.jpg", newImage)


def deskew(cvImage, filename:str):
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle, filename)


for img in imgs_path:
    image = cv2.imread(img, -1)
    cv2.imshow("oldImage", image)
    deskew(image, img)
    print(getSkewAngle(image))
cv2.waitKey(0)
