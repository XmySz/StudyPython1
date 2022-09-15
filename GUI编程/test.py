import os
import tkinter as tk
import tkinter.filedialog  # 文件选择对话框
from tkinter import scrolledtext    # 滚动文本
import time
import cv2
import easyocr
import warnings
warnings.filterwarnings("ignore", category=UserWarning) # 忽略用户警告

root = tk.Tk()
root.title("SVS文件自动重命名系统")
root.geometry("800x800")
root.config(background="#f5f5dc")   # 米白色
reader = easyocr.Reader(["en", ])
dirname = []


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


def choose_file():
    """
        文件选择器，返回所选择的文件的字符串/字符串元组形式
    :return:
    """
    # filename = tk.filedialog.askopenfilename()  # 选择文件
    # filename = tk.filedialog.askopenfilenames()   # 选择多个文件
    dirpath = tk.filedialog.askdirectory(title="选择目录") # 选择目录
    var.set(dirpath)
    time.sleep(2)
    dirname.append(dirpath)
    lb = tk.Label(root, font=("Courier", 12, "bold"), bg="#f5f5dc", text="正在处理中.......，此窗口可以关闭了")
    lb.place(x=230, y=165)


var = tk.StringVar()
btn = tk.Button(root, text="选择要更改文件名的目录/文件", font=("Courier", 12), command=choose_file,)
btn.place(x=300, y=100)
entry = tk.Entry(root, width=50, font=("Courier", 12), textvariable=var)
entry.place(x=230, y=135)


root.mainloop()

img_paths = [os.path.join(dirname[0], img) for img in os.listdir(dirname[0])]
for i, img_path in enumerate(img_paths):
    # im_label = tifffile.imread(img_path, series=-2)  # im_label为np对象
    print("正在处理{}, {}/{}".format(img_path, i+1, len(img_paths)))
    im_label = cv2.imread(img_path)
    rotated = is_rotated(im_label)
    result = reader.readtext(rotated, detail=0)
    new_name = extract(result) + ".svs"
    if '/' in new_name or '(' in new_name:
        continue
    print("文件名将从{}变更到{}".format(img_path, os.path.join(dirname[0], new_name)))
    print("------------------------------------------------------------------------")
    # os.rename(img_path, os.path.join(dirname[0], new_name))

