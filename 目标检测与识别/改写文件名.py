import os
import easyocr
import tifffile
import cv2

root = "/home/sci/zyn/test"
reader = easyocr.Reader(["en", ])
img_paths = [os.path.join(root, img) for img in os.listdir(root)]


def extract(result: list):
    """
        根据预测的结果提取有效的信息
    :param result: ['{1/]+cEB', '" J@A', '20194426', '5', '0']
    :return: 20194426-5-0
    """
    try:
        new_result = result[2:]
        part1 = max(new_result, key=lambda x: len(x))
        new_result.remove(part1)
        # 本部分负责去除第一部分末尾的乱码
        l = [i for i in part1]
        index = 0
        for i in range(len(l) - 1, -1, -1):
            if l[i].isdigit():
                index = i
                break
        part1 = part1[:index + 1]
        for s in new_result:
            part1 += ('-' + s)
        return part1
    except:
        return "error"


def is_rotated(img):
    """
        根据预测结果判断是否需要旋转
    :param img:np对象
    :return:原图或者旋转后的np对象
    """
    global reader
    result = reader.readtext(img, detail=0)
    if len([i for i in result if len(i) > 3]) == 0:  # 如果预测结果的每一项长度都小于3那就说明需要旋转
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 90, 1)  # 旋转90度变为正向
        img = cv2.warpAffine(img, M, (w, h))
    else:
        pass
    return img


for i, img_path in enumerate(img_paths):
    # im_label = tifffile.imread(img_path, series=-2)  # im_label为np对象
    im_label = cv2.imread(img_path)
    rotated = is_rotated(im_label)
    result = reader.readtext(rotated, detail=0)
    new_name = extract(result) + ".svs"
    if '/' in new_name or '(' in new_name:
        continue
    os.rename(img_path, os.path.join(root, new_name))
