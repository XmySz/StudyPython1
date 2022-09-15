"""
    python文件选择对话框
            filedialog.askopenfilename(***options)  自动打开选取窗口，手动选择一个文件，返回文件路径，类型为字符串。
            filedialog.askopenfilenames(**options)  同时选择多个文件，返回一个元组，包括所有选择文件的路径。
            filedialog.asksaveasfile(**options)     选择文件存储路径并命名
            filedialog.askdirectory(**options)      选择一个文件夹，返回文件夹路径。
            可选参数 **options:
                title --指定文件对话框的标题栏文本。
                defaultextension --指定文件的后缀，例如：defaultextension=’.jpg’，那么当用户输入一个文件名’Python’的时候，文件名会自动添加后缀为’Python.jpg’
                filetypes --指定筛选文件类型的下拉菜单选项，该选项的值是由二元组构成的列表，每个二元组是由（类型名，后缀）构成
                initialdir --指定打开保存文件的默认路径，默认路径是当前文件夹。
                multiple --是否确定选择多个文件
"""
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()  # 创建一个Tkinter.Tk()实例
root.withdraw()  # 将Tkinter.Tk()实例隐藏

# 选择一个文件
file_path = filedialog.askopenfilename(title='请选择一个文件', initialdir=r'D:\冰川数据\物质平衡模型测试数据',
                                       filetypes=[(
                                           "文本文档", ".txt"), ('Excel', '.xls .xlsx'), ('All Files', ' *')],
                                       defaultextension='.tif')
print(file_path)

# 选择多个文件
file_paths = filedialog.askopenfilenames(title='请选择多个文件', initialdir=r'D:\冰川数据\物质平衡模型测试数据',
                                         filetypes=[(
                                             "文本文档", ".txt"), ('Excel', '.xls .xlsx'), ('All Files', ' *')])
print(file_paths)

# 选择文件存储路径
save_file = filedialog.asksaveasfile(title='请选择文件存储路径', initialdir=r'D:\冰川数据\物质平衡模型测试数据',
                                     filetypes=[(
                                         "文本文档", ".txt"), ('Excel', '.xls .xlsx'), ('All Files', ' *')],
                                     defaultextension='.tif')
print(save_file)

# 选择要处理的文件的文件夹
dir_path = filedialog.askdirectory(title='选择影像存放的位置！', initialdir=r'D:\冰川数据\物质平衡模型测试数据')
print(dir_path)

