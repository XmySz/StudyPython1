"""
      该模块提供表示文件系统路径的类，其语义适用于不同的操作系统。路径类被分为提供纯计算操作而没有 I/O 的 纯路径，
    以及从纯路径继承而来但提供 I/O 操作的 具体路径。
    详细查看：https://zhuanlan.zhihu.com/p/139783331
"""
from pathlib import Path

print(Path.cwd())   # 获取工作目录
print(Path.home())  # 获取home目录
print(Path(__file__))   # 获取当前文件的绝对路径

file = Path("Path学习.py")

# 获取文件的信息
print(file.stat())
print(file.stat().st_size)
print(file.stat().st_atime)
print(file.stat().st_ctime)
print(file.stat().st_mtime)

# 获取路径组成部分
print(file.name)    # 文件名包括后缀
print(file.stem)    # 文件名不包括后缀
print(file.suffix)  # 文件的后缀
print(file.parent)  # 父级目录
print(file.anchor)  # 目录的锚
