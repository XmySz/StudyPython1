import csv

# csv 模块中的 writer 类可用于读写序列化的数据
# 操作文件对象时，需要添加newline参数逐行写入，否则会出现空行现象
with open('.\data\eggs.csv', 'w', newline='') as csvfile:   # 写
    spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|')  # delimiter 指定分隔符，默认为逗号，这里指定为空格
                                                                    # quotechar 表示引用符,当一段话中出现分隔符的时候，用引用符将这句话括起来，以能排除歧义
    spamwriter.writerow(['www.biancheng.net'] * 5 + ['how are you'])# writerow 单行写入，列表格式传入数据
    spamwriter.writerow(['hello world', 'web site', 'www.biancheng.net'])
    spamwriter.writerows([('hello','world'),('i','love','you')])    # 同时写入多行数据，需要使用 writerrows() 方法

    # 也可使用 DictWriter 类以字典的形式读写数据
    # 构建字段名称，也就是key
    fieldnames = ['first_name', 'last_name']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()    # 写入字段名，当做表头
    # 多行写入
    writer.writerows([{'first_name': 'Baked', 'last_name': 'Beans'}, {'first_name': 'Lovely', 'last_name': 'Spam'}])
    # 单行写入
    writer.writerow({'first_name': 'Wonderful', 'last_name': 'Spam'})


# csv 模块中的 reader 类和  DictReader 类用于读取文件中的数据
with open('.\data\eggs.csv', 'r', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        print(', '.join(row))

    reader = csv.DictReader(csvfile)
    for row in reader:
        print(row['first_name'], row['last_name'])