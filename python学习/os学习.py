"""
    os模块是python标准库中访问操作系统的模块，可以适应于不同的操作系统平台，
    快速完成文件、目录的相关操作，如查找文件的目录，为文件创建新目录等
        os.name         
        os.curdir   指代当前目录，也可以用.表示
        os.pardir   指代当前目录的上一级目录,也可以用..表示
        os.sep      返回路径名分隔符,可以是'//',也可以是'\'
    shutil模块可以提供方便的移动、复制和删除文件/文件夹等高级文件操作
"""
import os
import shutil

path = '/home/sci/zyn/test'

# os.chdir(path)    # 改变当前工作目录到path
# os.mkdir("test",) # 创建文件夹
# os.remove(path)   # 删除一个路径为path的文件
# os.unlink(src)    # 删除文件
# os.rmdir(path)    # 删除指定的 空 文件夹，非空会报错
# os.rename(scr, dst)   # 重命名文件或目录从src到dst
# os.startfile(path)    # 打开一个文件
print(os.getcwd())  # 返回当前的工作目录
print(os.name)  # 返回操作系统名称,如果是posix，说明系统是Linux、Unix或Mac OS，如果是nt，就是Windows系统。
print(os.uname())  # 返回操作系统的详细信息，windows上不可用
print(os.environ)  # 返回在操作系统中定义的环境变量
print(os.listdir(path))  # 返回某个目录下所有的文件名:list

print(os.path.join(path, os.listdir(path)[0]))  # 链接目录
print(os.path.split(os.path.join(path, os.listdir(path)[0])))  # 分割目录和文件:(目录，文件)
print(os.path.exists(path))  # 判断某个路径或者文件是否存在:bool
print(os.path.isdir(path))  # 判断某个路径是否是目录
print(os.path.isfile(path))  # 判断某个路径是否是文件
print(os.path.splitext(os.listdir(path)[0]))  # 分割路径，返回路径名和文件扩展名的元组
print(os.path.basename(os.path.join(path, os.listdir(path)[0])))    # 返回路径的尾部
print(os.path.abspath(path))    # 返回绝对路径


# os.walk()方法用于通过在目录树中游走输出在目录中的文件名，向上或者向下。
for root, dirs, files in os.walk("./"): # 参数分别为要遍历的目录的地址， topdown参数为是否有优先遍历top目录，或是优先遍历top的子目录
    print("root:", root)    # 返回一个三元组，第一个为top目录
    print("dirs:", dirs)    # 第二个为目录下的子目录名称列表
    print("files:", files)  # 第三个为目录下的文件列表


# shutil.copyfile(source, destination)    # 复制文件
# shutil.copytree(source, destination)    # 复制文件夹，目的文件夹不存在会自动创建
# shutil.move(source, destination)    # 移动文件
# shutil.rmtree(src)    # 删除文件夹，必须非空
# shutil.copy()     # 用于将源文件的内容复制到目标文件或目录。它还保留了文件的权限模式，但不保留文件的其他元数据，例如文件的创建和修改时间。
# shutil.copy2()    # 将源文件的内容复制到目标文件或目录。此方法与shutil.copy()方法相同，但它也尝试保留文件的元数据。

