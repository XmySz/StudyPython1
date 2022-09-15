from tkinter import *  # 导入模块

top = Tk()  # 设置窗口
sb = Scrollbar(top)  # 设置窗口滚动条
sb.pack(side=RIGHT, fill=Y)  # 设置窗口滚动条位置

mylist = Listbox(top, yscrollcommand=sb.set)  # 创建列表框

# 当Listbox组件的可视范围发生改变的时候,Listbox组件通过调用set()方法通知Scrollbar组件,而当用户操纵滚动条时,就自动调用Listbox组件的yview方法

# 添加水平滚动条方法跟上边一样,只是将yscrollcommand改为xscrollcommand,yview改成xview即可

for line in range(30):
    mylist.insert(END, "Number " + str(line))  # 设置范围

mylist.pack(side=LEFT)
sb.config(command=mylist.yview)


mainloop()
