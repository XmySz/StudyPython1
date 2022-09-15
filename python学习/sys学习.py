"""
    sys模块用于提供对Python解释器相关的操作.
    当我们使用 import 语句的时候，Python 解释器是怎样找到对应的文件的呢？
        这就涉及到Python的搜索路径，搜索路径是由一系列目录名组成的，Python解释器就依次从这些目录中去寻找所引入的模块。
    这看起来很像环境变量，事实上，也可以通过定义环境变量的方式来确定搜索路径。搜索路径是在Python编译或安装的时候确定的，安装新的库应该也会修改。
    搜索路径被存储在sys模块中的path变量`
"""
import sys

print(sys.path)         # 返回模块的搜索路径，初始化时使用PYTHONPATH环境变量的值
sys.path.append("..")   # 添加搜索路径
print(sys.argv)         # 命令行参数List，第一个元素是程序本身路径
print(sys.version)      # 获取Python解释程序的版本信息
print(sys.maxsize)      # 最大容量
print(sys.platform)     # 操作系统平台名称
si = sys.stdin          # 输入相关
so = sys.stdout         # 输出相关

sys.exit(1)             # 退出程序，正常退出时exit(0)

