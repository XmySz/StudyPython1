"""
    messagebox 模块用于在 python 应用程序中显示消息框。根据应用程序的要求，有各种用于显示相关消息的功能。
"""
from tkinter import *
from tkinter import messagebox
from tkinter.messagebox import *

top = Tk()
top.geometry("400x200")

# 消息框
showinfo("温馨提示", "内容")  # 提示消息框
showwarning('警告', '明日有大雨')  # 警告消息框
showerror('错误', '出错了')  # 错误消息框

# 对话框
askquestion("确认", "你确定吗？")  # 此方法用于向用户提出一些问题，可以回答是或否。
askokcancel("重定向", "将您重定向到 www.javatpoint.com")  # 此方法用于确认用户对某些应用程序活动的操作。
messagebox.askyesno("应用程序", "知道了吗？")  # 此方法用于向用户询问某些操作，用户可以回答是或否。
askretrycancel("application", "再试一次？")  # 此方法用于询问用户是否再次执行特定任务。

top.mainloop()
