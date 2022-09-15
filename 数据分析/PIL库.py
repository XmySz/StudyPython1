import PIL.Image as Image
import numpy as np

img_path = "/home/sci/zyn/test/2000807-6-EA-DX1.jpg"

img = Image.open(img_path)  # 读取一张图片，参数为图片路径，返回的是img格式，可以转化为ndarray对象
print(type(img))
print(np.asarray(img).shape)    # !!!注意PIL读取出来的图片换转为numpy格式和直接使用cv读取的图片在像素点并不是完全一致
img.show()  # 展示一张图片
# img.save()   保存图片,参数为文件名

# 查看图片的相关信息
print(img.size)  # 尺寸
print(img.mode)  # 模式
print(img.format)  # 格式

# 更改图像的格式
img.crop((0, 0, 200, 200))  # 裁剪图片，参数为四元组
img.resize((200, 200))  # 调整图片尺寸
img.rotate(45)  # 旋转图像


# 图像的颜色变化、
"""
    mode    描述
    1       1位像素，黑和白，存成8位的像素
    L       8位像素，黑白
    RGB     3*8位像素，真彩
    CMYK    4*8位像素，颜色隔离
    I       32位整型像素
    F       32位浮点像素    
"""
img.convert(mode='L').show()

