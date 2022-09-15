import csv
import random
import re
import time
from urllib import request
from urllib import parse  # 对url进行编码
from python爬虫.ua_info import ua_list


def _simple_urlopen():
    response = request.urlopen('http://www.baidu.com/')
    html = response.read().decode('utf-8')
    bytes = response.read()  # read()返回结果为 bytes数据类型
    string = response.read().decode()  # decode()将字节串转换为 string类型
    url = response.geturl()  # 返回响应对象的URL地址
    code = response.getcode()  # 返回请求时的HTTP响应码

    # 字符串转换为字节码
    string.encode("utf-8")
    # 字节码转换为字符串
    bytes.decode("utf-8")

    print(response)
    print(html, type(html))
    print(bytes, type(bytes))
    print(url, type(url))
    print(code, type(code))


def _User_Agent():
    """
        重构爬虫UA信息,防止反爬第一步
    :return:
    """
    url = 'http://httpbin.org/get'
    # 重构请求头
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:65.0) Gecko/20100101 Firefox/65.0"
    }
    # 1.创建请求对象，包装ua信息
    req = request.Request(url=url, headers=headers)
    # 2.发送请求,获取响应对象
    res = request.urlopen(req)
    # 3.提取响应内容
    print(res.read().decode('utf-8'))


def _url_encode():
    """
        URL 之所以需要编码，是因为 URL 中的某些字符会引起歧义，
        比如 URL 查询参数中包含了”&”或者”%”就会造成服务器解析错误；再比如，URL 的编码格式采用的是 ASCII 码而非 Unicode 格式，
        这表明 URL 中不允许包含任何非 ASCII 字符（比如中文），否则就会造成 URL 解析错误。
        哪些字符需要编码，分为以下三种情况：
             ASCII 表中没有对应的可显示字符，例如，汉字。
            不安全字符，包括：# ”% <> [] {} | \ ^ ` 。
            部分保留字符，即 & / : ; = ? @ 。
        urlencode()	该方法实现了对 url地址的编码操作
        unquote() 	该方法将编码后的 url地址进行还原，被称为解码
    :return:
    """
    # 构建查询字典
    query_string = {
        'wd': '爬虫'
    }
    # 方式1 调用parse模块的urlencode()对字典进行编码
    result = parse.urlencode(query_string)
    url = 'http://www.baidu.com/s?{}'.format(result)
    print(url)

    # 方式2 也可以使用quote(string)方法对字符串编码
    url = 'http://www.baidu.com/s?wd={}'
    word = '你好'
    print(url.format(parse.quote(word)))

    # 调用paras的unquote进行解码
    str = '%E4%BD%A0%E5%A5%BD'
    print(parse.unquote(str))

    # url拼接的几种方法
    # 1、字符串相加
    baseurl = 'http://www.baidu.com/s?'
    params = 'wd=%E7%88%AC%E8%99%AB'
    url = baseurl + params

    # 2、字符串格式化（占位符）
    params = 'wd=%E7%88%AC%E8%99%AB'
    url = 'http://www.baidu.com/s?%s' % params

    # 3、format()方法
    url = 'http://www.baidu.com/s?{}'
    params = 'wd=%E7%88%AC%E8%99%AB'
    url = url.format(params)


def _practice():
    """
    一个网站爬取的简单实战
    :return:
    """

    # 1.拼接地址
    def get_url(word):
        url = 'http://www.baidu.com/s?{}'
        params = parse.urlencode({'wd': word})
        url = url.format(params)
        return url

    # 2.发送爬取请求,保存到本地
    def request_url(url, filename):
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:6.0) Gecko/20100101 Firefox/6.0'}
        req = request.Request(url=url, headers=headers)
        response = request.urlopen(req)
        html = response.read().decode('utf-8')
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)

    word = input("请输入搜索内容:")
    url = get_url(word)
    filename = word + '.html'
    request_url(url, filename)


def _baidutieba():
    """
    本次爬取百度贴吧内容
        url为 https://tieba.baidu.com/f?kw=p&ie=utf-8&pn=150
    :return:
    """

    class TiebaSpider():
        def __init__(self):
            self.url = 'http://tieba.baidu.com/f?{}'

        # 1.请求函数,得到页面
        def get_url(self, url):
            req = request.Request(url=url, headers={"User-Agent": random.choice(ua_list)})
            res = request.urlopen(req)
            html = res.read().decode('gbk', 'ignore')
            return html

        # 2.解析函数
        def parse_html(self):
            pass

        # 3.保存文件函数
        def save_html(self, filename, html):
            with open(filename, 'w') as f:
                f.write(html)

        # 4.入口
        def run(self):
            name = input('输入贴吧名:')
            begin = int(input('输入起始页:'))
            end = int(input('输入终止页:'))
            for page in range(begin, end + 1):
                pn = (page - 1) * 50
                params = {
                    'kw': name,
                    'pn': str(pn),
                }
                # 拼接url
                params = parse.urlencode(params)
                url = self.url.format(params)
                # 发请求
                html = self.get_url(url)
                # 定义路径
                filename = './data/{}-{}页.html'.format(name, page)
                self.save_html(filename, html)
                # 提示信息
                print("第%d个页面抓取成功!" % page)
                # 每次爬取一个页面休息3s
                time.sleep(random.randint(1, 3))

    begin = time.time()
    spider = TiebaSpider()
    spider.run()
    end = time.time()
    # 查看程序执行时间
    print("执行时间:%0.2fs" % (end - begin))


def _maoyantop100():
    """
    爬取猫眼top100影片信息,包括电影名称,上映时间,主演名字

    :return:
    """

    class MaoyanSpider(object):
        # 1.定义初始页面
        def __init__(self):
            self.url = 'https://www.maoyan.com/board/4?offset={}'

        # 2.请求函数
        def get_html(self, url):
            headers = {'User-Agent': random.choice(ua_list)}
            req = request.Request(url=url, headers=headers)
            res = request.urlopen(req)
            html = res.read().decode()
            # 调用解析函数
            self.parse_html(html)

        # 3.解析函数
        def parse_html(self, html):
            # 正则表达式
            regex = '<div class="movie-item-info">.*?title="(.*?)".*?class="star">(.*?)</p>.*?releasetime">(.*?)</p>'
            # 生成正则表达式对象
            pattern = re.compile(regex, re.S)
            r_list = pattern.findall(html)
            self.save_html(r_list)

        # 4.保存数据函数,使用csv
        def save_html(self, r_list):
            # 生成文件对象
            with open('.\data\maoyan.csv', 'a', newline='', encoding="utf-8") as f:
                # 生成csv操作对象
                writer = csv.writer(f)
                # 整理数据
                for r in r_list:
                    name = r[0].strip()
                    star = r[1].strip()[3:]
                    time = r[2].strip()[5:15]
                    L = [name, star, time]
                    writer.writerow(L)
                    print(name, time, star)

        # 5.主函数
        def run(self):
            # 抓取第一页的数据
            for offset in range(0, 11, 10):
                url = self.url.format(offset)
                self.get_html(url)
                time.sleep(random.uniform(2, 3))
    begin = time.time()
    try:
        spider = MaoyanSpider()
        spider.run()
    except Exception as e:
        print("异常:", e)
    end = time.time()
    print("爬取消耗时间为%0.2f s"%(end-begin))