import re  # 拥有python语言全部的正则表达式功能

# compile函数根据一个模式字符串和可选的标志参数生成一个正则表达式对象。该对象拥有一系列方法用于正则表达式匹配和替换。
# re模块也提供了与这些方法功能完全一致的函数，这些函数使用一个模式字符串做为它们的第一个参数。
"""
    正则表达式：
        格式:         /pattern/修饰符                   
        修饰符:
            i       :不区分大小写
            g       :全局匹配
            m       :多行匹配，使边界字符^和$匹配每一行的开头和结尾,记住是多行,而不是整个字符串的开头和结尾。
            s       :特殊字符圆点.中包含换行符\n,默认情况下的圆点.是匹配除换行符\n之外的任何字符,加上s修饰符之后,.中包含换行符\n。
        元字符：
            *       :匹配前面的0次或多次
            ?       :匹配前面的0次或1次，注意同样也有非贪婪模式的含义，当且仅当紧跟在任何使用闭合操作符的匹配后面
            +       :匹配前面的1次或多次
            ^       :匹配字符串开始的位置
            ＄      :匹配字符串结束的位置
            |       :逻辑或
            {n}     :匹配确定的n次
            {n,}    :匹配最少n次
            {n,m}   :匹配最少n次最多m次
            某个限定符后紧跟？   :转变为非贪婪匹配模式(尽可能少的匹配)
            .       :匹配除了\n和\r以外的任意字符
            ()          :分组,匹配子组,提取信息,需要哪个特定信息的时候，就可以通过分组(也就是加括号)的方式获得。
            (pattern)   :匹配这个pattern并获取
            (?:pattern) :匹配但是不获取结果
            (?=pattern) :正向肯定预查,在任何匹配pattern的字符串开始处匹配查找字符串。例如Windows(?=95|98|NT|2000)"能匹配"Windows2000"中的"Windows"，但不能匹配"Windows3.1"中的"Windows"。 
            (?!pattern) :正向否定预查,预查在任何不匹配pattern的字符串开始处匹配查找字符串。这是一个非获取匹配，也就是说，该匹配不需要获取供以后使用。例如"Windows(?!95|98|NT|2000)"能匹配"Windows3.1"中的"Windows"，但不能匹配"Windows2000"中的"Windows"
            (?<=pattern):反向肯定预查,与正向相反，朝后匹配
            (?<=pattern):反向否定预查,与反向相反，朝后匹配
            [xyz]       :可以匹配所包含的任一字符
            [^xyz]      :匹配未包含的任意字符
            [a-z]       :匹配指定范围的任意字符
            [^a-z]      :匹配非指定范围的任意字符
            \b          :匹配一个单词边界,例如，'er\b' 可以匹配"never" 中的 'er'，但不能匹配 "verb" 中的 'er'。
            \       :匹配一个转义字符
            \B          :匹配一个非单词边界
            \cx         :匹配由x指明的控制字符
            \d          :匹配一个数字字符
            \D          :匹配一个非数字字符
            \s          :匹配任何空白字符，空格制表或换页
            \S          :匹配任何非空白字符
            \w          :匹配数字字母下划线
            \W          :匹配非数字字母下划线
            <>          :匹配h1标签开始和关闭之间的所有内容
    # .* 表示任意匹配除换行符（\n、\r）之外的任何单个或多个字符
    # (.*?) 表示"非贪婪"模式，只保存第一个匹配到的子串              
            
"""

"""
    使用圆括号指定分组：
        有些时候，我们可能会对之前匹配成功的数据更感兴趣。我们不仅想要知道整个字符串是否匹配我们的标准，而且想要知道能否提取任何已经成功匹配的特定字符串或者子字符串。
     re.compile(pattern[, flags])
        编译正则表达式
        关于为什么要编译?
            调用一个代码对象而不是一个字符串，性能上会有明显提升。这是由于对于前者而言，编译过程不会重复执行。
        换句话说，使用预编译的代码对象比直接使用字符串要快，因为解释器在执行字符串形式的代码前都必须把字符串编译成代码对象。
        在模式匹配发生之前，正则表达式模式必须编译成正则表达式对象。由于正则表达式在执行过程中将进行多次比较操作，
        因此强烈建议使用预编译。而且，既然正则表达式的编译是必需的，那么使用预编译来提升执行性能无疑是明智之举。
        
    group()与groups():
        当处理正则表达式时，除了正则表达式对象之外，还有另一个对象类型：匹配对象。
        这些是成功调用match()或者search()返回的对象。匹配对象有两个主要的方法：group()和groups()。
        group()要么返回整个匹配对象，要么根据要求返回特定子组。
        groups()则仅返回一个包含唯一或者全部子组的元组。如果没有子组的要求，那么当group()仍然返回整个匹配时，groups()返回一个空元组。

    
    re.match(pattern, string, flags=0)
        pattern:匹配的正则表达式
        string:要匹配的字符串
        flags:修饰符
            match()函数试图从字符串的"起始部分"对模式进行匹配。如果匹配成功，就返回一个匹配对象；如果匹配失败，就返回 None，
        匹配对象的 group()方法能够用于显示那个成功的匹配。
        
    re.search(pattern, string, flags=0)
        在任意位置对给定正则表达式模式搜索第一次出现的匹配情况。如果搜索到成功的匹配，就会返回一个匹配对象；否则，返回None。

    re.findall(pattern, string, flags=0)
        在字符串中找到正则表达式所匹配的所有子串,返回一个列表。如果有多个匹配模式，则返回元组列表，如果没有找到匹配的，则返回空列表。

    re.finditer(pattern, string, flags=0)
        在字符串中找到正则表达式所匹配的所有子串，并把它们作为一个迭代器返回。
    
    re.sub(pattern, repl, string, count=0, flags=0)
        替换字符串中的匹配项
        repl：替换的字符串，可以是一个函数
        string：要被查找的字符串
        count：最大次数
    
    re.subn()
         subn()还返回一个表示替换的总数，替换后的字符串和表示替换总数的数字一起作为一个拥有两个元素的元组返回。
        
    re.split(pattern, string[, maxsplit=0, flags=0])
        split方法按照能够匹配的子串将字符串分割后返回列表
        
"""
# match的示例
line = """RegExr was created by gskinner.com, and is proudly hosted by Media Temple.
Edit the Expression & Text to see matches. Roll over matches or the expression for details. PCRE & JavaScript flavors of RegEx are supported. Validate your expression with Tests mode.
The side bar includes a Cheatsheet, full Reference, and Help. You can also Save & Share with the Community, and view patterns you create or favorite in My Patterns.
Explore results with the Tools below. Replace & List output custom results. Details lists capture groups. Explain describes your expression in plain English.
`1234567890-=~!@#$%^&*()_+qwertyuiop[]\asdfghjkl;'zxcvbnm,./{}|:"<>? \n\b\B\w\W\d\D
"""

pattern = "Reg"
matchObj = re.match(pattern, line, re.M | re.I)

if matchObj:
    print(matchObj.group())
    print(matchObj.group(0))
    print(matchObj.groups())
else:
    print("No match!!")

# serach的示例
searchObj = re.search(r'(.*) are (.*?) .*', line, re.M | re.I)

if searchObj:
    print("searchObj.group() : ", searchObj.group())
    print("searchObj.group(1) : ", searchObj.group(1))
    print("searchObj.group(2) : ", searchObj.group(2))
else:
    print("Nothing found!!")

# sub的示例
phone = "2004-959-559 # 这是一个电话号码"

# 删除注释
num = re.sub(r'#.*$', "", phone)
print("电话号码 : ", num)

# 移除非数字的内容
num = re.sub(r'\D', "", phone)
print("电话号码 : ", num)

# compile的示例
pattern = re.compile(r'\d+')  # 用于匹配至少一个数字
m = pattern.match('one12twothree34four')  # 查找头部，没有匹配
print(m)

print(re.search(r'e', 'bitethe dog'))
