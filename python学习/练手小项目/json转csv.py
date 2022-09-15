"""
    什么是json文件呢?
        参考json模块
    什么是csv文件呢?
        逗号分隔值（Comma-Separated Values，CSV，有时也称为字符分隔值，因为分隔字符也可以不是逗号）.
        其文件以纯文本形式存储表格数据（数字和文本）。纯文本意味着该文件是一个字符序列，不含必须像二进制数字那样被解读的数据。
        CSV文件由任意数目的记录组成，记录间以某种换行符分隔；每条记录由字段组成，字段间的分隔符是其它字符或字符串，
        最常见的是逗号或制表符。通常，所有记录都有完全相同的字段序列。
"""
import json

try:
    with open('../data/input.json') as f:
        data = json.loads(f.read())

    output = ",".join([*data[0]])  # *用来拆包,放在list,tuple前会它们拆成单个元素,放在字典前会拆开键,**放在字典前拆开字典的每一项

    for obj in data:
        output += f'\n{obj["Name"]},{obj["age"]},{obj["birthyear"]}'  # f-string的大括号{}可以填入表达式或调用函数

    with open('../data/output.csv', 'w') as f:
        f.write(output)

except Exception as ex:
    print(f'Error: {str(ex)}')
