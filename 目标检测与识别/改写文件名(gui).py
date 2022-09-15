import os
from tkinter import *
import tkinter as tk
import tkinter.filedialog  # 文件选择对话框
from tkinter.messagebox import showinfo
from tkinter import scrolledtext, ttk  # 滚动文本
import time
import cv2
import easyocr
import warnings

warnings.filterwarnings("ignore", category=UserWarning)  # 忽略用户警告

root = Tk()
root.title("SVS文件自动重命名系统")
root.geometry("700x700")
root.config(background="#D8E9E6")
reader = easyocr.Reader(["en", ])


def process_part2(s: str):
    """
        变废为宝
    :param s:
    :return:
    """
    if len(s) > 6 and s[-6] == '(' and s[-1] == ')':
        return "CD3(2GV6)"
    elif len(s) <= 3 and not s.isdigit() and s[-1] == '8':
        return "CD8"
    else:
        return s


def extract(result: list):
    """
        根据预测的结果提取有效的信息并组合
    :param result: ['{1/]+cEB', '" J@A', '20194426', '5', '0']
    :return: 20194426-5-0
    """
    part1, part2, part3, part4 = "", "", "", ""
    new_result = result.copy()
    try:
        for s in result:  # 第一次循环寻找纯数字的长字符串，满足长度为7或者8且为纯数字
            if (len(s) == 7 or len(s) == 8) and s.isdigit():
                part1 = s
                result.remove(part1)
                break

        for s in result:  # 第二次循环找数字字母组合，满足以下几种情况
            s = process_part2(s)
            if s in ["EA", "HE", "CD3", "CD8", "CD3(2GV6)", "CD3(2GV6) "]:
                if '(' in s:
                    part2 = s[:3]
                else:
                    part2 = s
                result.remove(s)
                break

        for s in result:  # 第三次循环寻找一个纯数字的短字符串，长度小于3
            if (len(s) <= 2 and s.isdigit()) or s == "R2" or s == "B":
                part3 = s
                result.remove(part3)
                break

        for s in result:  # 第四次循环也是寻找一个纯数字的短字符串，长度小于3
            if len(s) <= 2 and s.isdigit():
                part4 = s
                result.remove(part4)
                break
        if part4 != "":
            return part1 + "-" + part2 + "-" + part3 + "-" + part4
        elif part3 != "":
            return part1 + "-" + part2 + "-" + part3
        else:
            return part1 + "-" + part2 + "-7"
    except:
        return "no-correct"


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
    dirpath = tk.filedialog.askdirectory(title="选择目录", )  # 选择目录
    var.set(dirpath)
    time.sleep(1)

    textw = scrolledtext.ScrolledText(root, width=70, height=30, )
    textw.place(x=90, y=130)
    textw.config(background="#f5f5dc", foreground="black", font='Courier 12 bold', wrap='word')
    textw.insert(END, "开始处理咯....")

    img_paths = [os.path.join(dirpath, img) for img in os.listdir(dirpath)]

    progressbarOne = ttk.Progressbar(root, length=400)  # 添加进度条的支持
    progressbarOne.place(x=160, y=650)
    # 进度值最大值
    progressbarOne['maximum'] = len(img_paths)
    # 进度值初始值
    progressbarOne['value'] = 0

    for i, img_path in enumerate(img_paths):
        # im_label = tifffile.imread(img_path, series=-2)  # im_label为np对象
        textw.insert(END, "正在处理{}, {}/{}\n".format(img_path, i + 1, len(img_paths)))
        im_label = cv2.imread(img_path)
        rotated = is_rotated(im_label)
        result = reader.readtext(rotated, detail=0)
        new_name = extract(result) + ".svs"
        if '/' in new_name or '(' in new_name:
            continue
        textw.insert(END, "文件名将从{}变更到{}\n".format(img_path, os.path.join(dirpath, new_name)))
        textw.insert(END, "----------------------------------------------------------------------\n")
        textw.see(END)  # 文本框始终显示最新的处理结果
        progressbarOne['value'] = i + 1
        root.update()  # 插入一次更新一次
        # os.rename(img_path, os.path.join(dirname[0], new_name))
    textw.insert(END, "处理完毕...")
    root.update()
    showinfo("提示", "全部文件处理完毕！")


var = tk.StringVar()
btn = tk.Button(root, text="选择要更改文件名的目录", font=("Courier", 12), command=choose_file, )
btn.place(x=250, y=50)
entry = tk.Entry(root, width=50, font=("Courier", 12), textvariable=var)
entry.place(x=160, y=90)

root.mainloop()
