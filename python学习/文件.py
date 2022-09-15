"""
    文件打开方式
        1.'r':读取，打开文件并读取，文件不存在则报错
        2.'a':追加，打开供追加的文件，不存在则创建该文件，指针在末尾
        3.'w':写入，打开文件进行写入，不存在则创建该文件，指针在开头(全覆盖)
        4.'x':创建，创建指定文件，不存在则返回错误
    指定文件处理模式
        1.'t':文本模式
        2.'b':二进制模式
    +:代表可读可写！
"""


with open('data/test.txt', 'r') as f:  # 使用with方式打开文件，可以在文件使用完毕后自动关闭
    print(f.read())

with open('data/t1.png', 'rb') as f:  # 使用rb来打开二进制文件，音乐视频，图片，等等
    print(f.read())

with open('data/test1.txt', 'r', encoding='gbk') as f:  # 要读取非UTF-8编码的文本文件，需要给open()函数传入encoding参数
    print(f.read())

with open('data/test.txt', 'r') as f:
    # print(f.read())                         # 默认读取文件全部内容,返回一个str
    # print(f.read(5))                        # 读取指定数量字符
    # print(f.readline())                     # 读取一行
    # print(f.readlines())                    # 读取所有行

    f1 = open('data/test1.txt', 'a')
    f2 = open('data/test2.txt', 'w')

    f1.write('123')
    f2.write('123')

f1.close()
f2.close()
# os.rename('ex1data1.txt', '第一个作业的数据')                 # 重命名文件
# os.remove('ex1data1.txt')                                   # 删除文件
# os.rmdir('文件夹文字')                                       # 删除文件夹
