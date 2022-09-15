import os
from time import sleep
from tkinter import *


class DirList(object):
    def __init__(self, initdir=None):
        self.top = Tk()   # 顶部窗口
        self.label = Label(self.top, text='Directory Lister v1.1')      # 第一个标签显示主标题和版本号
        self.label.pack()

        self.cwd = StringVar(self.top)     # 保存当前所在的目录名

        self.dir1 = Label(self.top, fg='blue', font=('Helvetica', 12, 'bold'))     # 显示当前的目录名
        self.dir1.pack()

        self.dirfm = Frame(self.top)    # 一个放置要列出的目录的文件列表的容器
        self.dirsb = Scrollbar(self.dirfm)  # 滚动条
        self.dirsb.pack(side=RIGHT, fill=Y) # 显示在右边，纵向
        self.dirs = Listbox(self.dirfm, height=15, width=50, yscrollcommand=self.dirsb.set)     # 列表框
        self.dirs.bind('<Double-1>', self.setDirAndGo)  # 绑定列表框与回调函数setDirAndGo
        self.dirsb.config(command=self.dirs.yview)  # 连接列表框与滚动条
        self.dirs.pack(side=LEFT, fill=BOTH)
        self.dirfm.pack()

        self.dirn = Entry(self.top, width=50, textvariable=self.cwd)    # 一个文本框，接受用户响应查询的目录
        self.dirn.bind('<Return>', self.doLS)   # 和函数doLS绑定
        self.dirn.pack()

        self.bfm = Frame(self.top)  # 定义一个按钮框架（容器）
        self.clr = Button(self.bfm, text='Clear', command=self.clrDir, bg='red', fg='white')    # 清空按钮
        self.ls = Button(self.bfm, text='List Directory', command=self.doLS, bg='green', fg='white')  # go按钮
        self.quit = Button(self.bfm, text='QUIT', command=self.top.quit, bg='red', fg='white')  # 退出按钮
        self.clr.pack(side=LEFT)
        self.ls.pack(side=LEFT)
        self.quit.pack(side=LEFT)
        self.bfm.pack()

        if initdir:         # 初始化当前目录为起点
            self.cwd.set(os.curdir)
            self.doLS()

    def clrDir(self, ev=None):
        self.cwd.set('')

    def setDirAndGo(self, ev=None):
        self.last = self.cwd.get()
        self.dirs.config(bg='red')
        check = self.dirs.get(self.dirs.curselection())
        if not check:
            check = os.curdir
        self.cwd.set(check)
        self.doLS()

    def doLS(self, ev=None):
        error = ''
        tdir = self.cwd.get()
        if not tdir:
            tdir = os.curdir
        if not os.path.exists(tdir):
            error = tdir + ': no such file'
        elif not os.path.isdir(tdir):
            error = tdir + ': not a directory'
        if error:
            self.cwd.set(error)
            self.top.update()
            sleep(2)
            if not (hasattr(self, 'last') and self.last):
                self.last = os.curdir
            self.cwd.set(self.last)
            self.dirs.config(bg='LightSkyBlue')
            self.top.update()
            return

        self.cwd.set('FETCHING DIRECTORY CONTENTS...')
        self.top.update()
        dirlist = os.listdir(tdir)
        dirlist.sort()
        os.chdir(tdir)
        self.dir1.config(text=os.getcwd())
        self.dirs.delete(0, END)
        self.dirs.insert(END, os.curdir)
        self.dirs.insert(END, os.pardir)
        for eachFile in dirlist:
            self.dirs.insert(END, eachFile)
        self.cwd.set(os.curdir)
        self.dirs.config(bg='LightSkyBlue')


def main():
    d = DirList(os.curdir)
    mainloop()


if __name__ == '__main__':
    main()
