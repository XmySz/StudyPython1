"""
    scikit-image是基于scipy的一款图像处理包，它将图片作为numpy数组进行处理,
    skimage库的全称scikit-image Scikit， 是对scipy.ndimage进行了扩展，提供了更多的图片处理功能。
"""

from skimage import io, data, feature

# io模块提供了图片的读取，显示和保存等操作
img = io.imread("./data/img.png")  # 返回np对象
io.imshow(img)
io.show()  # 显示图片
io.imsave("./data/new_img.png", img)  # 保存图片

# 获取图片信息
print(type(img))  # 类型
print(img.shape)  # 形状
print(img.shape[0])  # 图片宽度
print(img.shape[1])  # 图片高度
print(img.shape[2])  # 图片通道数
print(img.size)  # 显示总像素个数
print(img.max())  # 最大像素值
print(img.min())  # 最小像素值
print(img.mean())  # 像素平均值

# 图像处理
