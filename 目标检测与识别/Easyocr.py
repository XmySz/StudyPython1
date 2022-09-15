import easyocr
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)  # 忽略用户警告


def combiner(l: list):
    """
        合并已经提取出来的有效信息。
    :param l: 形如['CD8', '2052290', '10']
    :return:  2052290-10-CD8-DX1
    """
    part3_pre = ["CD3", "CD8", "HE", "EA", "D3", "D8"]
    part1: str = max(l, key=lambda x: len(x))
    l.remove(part1)
    try:
        part3 = [i for i in l if i in part3_pre][0]
        l.remove(part3)
        part2 = l[0]
        return part1 + "-" + part2 + "-" + part3 + "-" + "DX1"
    except:
        pass


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


reader = easyocr.Reader(['en'], )
root = "/home/sci/zyn/WSI-label"
imgs = os.listdir(root)  # 所有图片名
imgs_path = [os.path.join(root, p) for p in imgs]  # 所有图片的绝对路径名
split_imgs = [os.path.splitext(i)[0] for i in imgs]  # 不带后缀的图片名， 形如 2004544-16-EA-DX1

results = []  # 预测的中间结果   形如['2014234', '13', 'CD3', '2020/5/20', 'CTAARER_', 'F0Fe #RFHN4']
for img_path in imgs_path:
    part1, part2, part3, part4 = "", "", "", ""
    result = reader.readtext(img_path, detail=0)
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
            print(part1 + "-" + part2 + "-" + part3 + "-" + part4, new_result)
        elif part3 != "":
            print(part1 + "-" + part2 + "-" + part3, new_result)
        else:
            print(part1 + "-" + part2 + "-7", new_result)
    except:
        pass

    # for i in range(len(result)):  # 第一次筛选,改写带括号的项
    #     if "(" in result[i]:
    #         result[i] = result[i][:result[i].find('(')]
    # for i in [s for s in result if "/" in s or "G" in s or (len(s) > 3 and not s.isdigit())]:  # 第二次筛选去掉日期和“GGH”和识别错误字符串
    #     result.remove(i)
    # results.append(result[:3])

# final = []  # 预测的最终结果
# for result in results:
#     final.append(combiner(result))
#
# sum = 0
# for i, j in zip(split_imgs, final):
#     print("真实文件名：{}\n预测文件名：{}".format(i, j))
#     print("---------------------------------------")
#     if i == j:
#         sum += 1
# print("中标数/总样本数：{}/{}".format(sum, 1216))

# x = reader.readtext("/home/sci/zyn/WSI-label/2000074-12-CD3-DX1.jpg")
# for i in x:
#     print(i)
