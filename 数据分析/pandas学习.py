import os
import numpy as np
import pandas as pd
"""
        pandas 是用于处理和分析数据的 Python 库。它基于一种叫作 DataFrame 的数据结构，这
    种数据结构模仿了 R 语言中的 DataFrame。简单来说，一个 pandas DataFrame 是一张表
    格，类似于 Excel 表格。pandas 中包含大量用于修改表格和操作表格的方法，尤其是可
    以像 SQL 一样对表格进行查询和连接。NumPy 要求数组中的所有元素类型必须完全相
    同，而 pandas 不是这样，每一列数据的类型可以互不相同（比如整型、日期、浮点数和字
    符串）。pandas 的另一个强大之处在于，它可以从许多文件格式和数据库中提取数据，如
    SQL、Excel 文件和逗号分隔值（CSV）文件。
"""
print(os.getcwd())
df = pd.DataFrame(
    {
        "Name": [
            "Braund, Mr. Owen Harris",
            "Allen, Mr. William Henry",
            "Bonnell, Miss. Elizabeth",
        ],
        "Age": [22, 35, 58],
        "Sex": ["male", "male", "female"],
    }
)

series = pd.Series([11,22,33], name="age")  # 单独一列，等价于DataFrame中的一列,选择单个列时，返回的对象是 pandas Series
print(f"""series:\n{series}\ndf['Age']:\n{df["Age"]}""")
print(f"df的一些统计信息:{df.describe()}")


# 读取和写入文件中的数据 read_*函数用于向 pandas 读取数据，而方法to_*用于存储数据。
house_tiny = pd.read_csv("./data/house_tiny.csv")
print(house_tiny)   # 查看全部
print(house_tiny.head(5))   # 查看前几行
print(house_tiny.tail(5))   # 查看后几行
print(house_tiny.dtypes)    # 查看每列的数据类型
print("摘要:\n".format(house_tiny.info()))  # 查看摘要
house_tiny.to_excel("house.xlsx", sheet_name="test", index=False)   # 将数据存储为 excel 文件。


# 选取DataFrame的子集,
ages = house_tiny["Price"]  # 选择单个列时，返回的对象是 pandas Series
number_price = house_tiny[["NumRooms", "Price"]]    # 选择多列
above_15w = house_tiny[house_tiny["Price"]>150000]  # 过滤某些不符合条件的行,可以使用|和&代替and/or
print(house_tiny["Price"] > 15000)  # 返回根据条件筛选后只包含True/False的Series对象
priceis15w = house_tiny[house_tiny["Price"].isin([106000])] # isin()条件函数为提供的列表中的每一行返回一个True值。
print(house_tiny.loc[house_tiny["Price"] > 150000, "Alley"])    # 使用行名和列名时选择特定的行和/或列
print(house_tiny[1:, :])   # 使用表格中的位置时选择特定的行和/或列