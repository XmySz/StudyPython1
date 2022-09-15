import cv2
from PIL import Image

"""
    imread返回的是ndarray格式,两个参数：
        第一个参数为文件路径，第二个参数为打开的模式：
        第二个参数为打开的模式:cv2.IMREAD_COLOR：彩色图片   1
                           cv2.IMREAD_GRAYSCAL:灰度图片  0
                           cv2.IMREAD_UNCHANGED:包括alpha通道的图片   -1     
"""
img = cv2.imread("./data/1.jpg")
print(type(img))
print(img.shape)

# 保存图片
cv2.imwrite("./data/2.jpg", img)    # 两个参数，分别为保存的文件名和要保存的图片
# 显示图片
cv2.imshow("test", img)     # 参数为名字和图片
# opencv转PIL.Image
pil_img = Image.fromarray(img)
# 延时函数
cv2.waitKey(0)  # 单位毫秒,搭配imshow函数一起使用


# 处理图像
cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)  # 图像彩色空间变换函数,参数1为要变换的图像，参数2为变换成的模式cv2.COLOR_BGR2GRAY:从bgr转成灰度图
                                                                                          # cv2.COLOR_BGR2HSV：从bgr转到hsv空间
cv2.GaussianBlur(img, (9, 9), 0)    # 高斯模糊
cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) # 阈值化

# 旋转图片
h, w = img.shape[:2]
center = h//2, w//2
M = cv2.getRotationMatrix2D(center, angle=45, scale=1)    # 制作用于旋转图像的变换矩阵M，参数分别为中心，角度，比例（缩放因子）
rotated_img = cv2.warpAffine(src=img, M=M, dsize=(h, w))   # 返回旋转后的图片


"""
    连通域处理函数 连通区域一般是指图像中具有相同像素值且位置相邻的前景像素点组成的图像区域。连通区域分析是指将图像中的各个连通区域找出并标记。
    参数1为输入的二值图像，参数2为可选4连通或者8连通
    返回值：num_labels  所有连通域的数量
          labels      图像中每一像素的标记
          stats       每一个标记的统计信息i，是一个5列的矩阵，每一行为连通区域的x，y，width，height，area
          centroids   连通域的中心点
"""
cv2.connectedComponentsWithStats(img, connectivity=4)





