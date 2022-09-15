"""
    glob模块可以使用Unix shell风格的通配符匹配符合特定格式的文件和文件夹，跟windows的文件搜索功能差不多。
    glob模块并非调用一个子shell实现搜索功能，而是在内部调用了os.listdir()和fnmatch.fnmatch()。
    glob支持的通配符:
        *   匹配0个或多个字符
        **  匹配所有文件，目录，子目录和子目录里的文件
        ？   匹配一个字符
        [1-9]  匹配指定范围内的字符
        [!1-9]  匹配不在指定范围内的字符
"""
import glob


"""
    第一个参数pathname为需要匹配的字符串。
    第二个参数代表递归调用，与特殊通配符“**”一同使用，默认为False。
    该函数返回一个符合条件的路径的字符串列表
"""
paths = glob.glob("/home/sci/zyn/WSI-label/*.jpg", recursive=False) # 注意！会破坏原来的文件顺序

for path in glob.iglob("/home/sci/zyn/WSI-label/*.jpg"):    # 与glob类似，但是返回一个迭代器，避免过高的内存占用
    print(path)