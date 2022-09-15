from tkinter import *

"""
    拖动滑块改变文字大小 
"""
top = Tk()  # 顶层窗口


def resize(ev=None):
    label.config(font='Helvetica -%d bold' % scale.get())


top.geometry('500x500')  # 调整顶部窗口大小

label = Label(top, text="Hello, World", font='Helvetica -12 bold')
label.pack(fill=X, expand=1)

scale = Scale(top, from_=10, to=80, orient=HORIZONTAL, command=resize)
scale.set(12)
scale.pack(fill=X, expand=1)

quit = Button(top, text="quit", command=top.quit, bg='red', fg='white')
quit.pack()

mainloop()  # 无限主循环，一般在最后一行
