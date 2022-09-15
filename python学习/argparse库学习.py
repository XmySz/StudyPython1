"""
    argparse模块主要用于处理Python命令行参数和选项，程序定义好所需参数后，该模块会通过sys.argv解析出那些参数；
    除此之外，argparse模块还会自动生成帮助和使用手册，并在用户给程序传入无效参数时报出错误信息。
"""
import sys
import argparse

# 创建解析对象
parse = argparse.ArgumentParser("test,test,test")
"""
    ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True)
    参数详解：
        prog：程序的名称，默认为sys.argv[0]
        usage: 描述程序用途的字符串
        description：在参数帮助文档之前显示的文本（默认值：无）
        epilog：在参数帮助文档之后显示的文本（默认值：无）
        ......
"""

# 添加参数
parse.add_argument(
    '-n', '--name', dest='rname', required=True, help='增加输出的名字'
)
parse.set_defaults(n="zyn") # 设置默认参数
"""
    add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
        name or flags：一个命名或者一个选项字符串的列表，例如：-f，--foo
        action：当参数在命令行中出现时使用的动作基本类型
        nargs：命令行参数应当消耗的数目
        const：被一些 action 和 nargs 选择所需求的常数
        default：当参数未在命令行中出现时使用的值
        type：命令行参数应当被转换成的类型
        choices：可用的参数的容器
        required：此命令行选项是否可省略
        help：一个选项作用的简单描述
        metavar：在使用方法消息中使用的参数值示例
        dest：被添加到 parse_args() 所返回对象上的属性名     
"""

# 解析参数对象获得解析对象
args = parse.parse_args()
"""
    parse_args(args=None, namespace=None)：
        args：要分析的字符串列表，默认取自 sys.argv
        namespace：命名空间
"""


name = args.rname
print("hello, ", name)