"""
    静态网页与动态网页
        静态网页：
            静态网页是标准的 HTML 文件，通过 GET 请求方法可以直接获取，文件的扩展名是.html、.htm等，
            网页中可以包含文本、图像、声音、FLASH 动画、客户端脚本和其他插件程序等。，且不需要连接后台数据库
            因此响应速度非常快。但静态网页更新比较麻烦，每次更新都需要重新加载整个网页。
        动态网页：
            动态网页指的是采用了动态网页技术的页面，比如
            AJAX（是指一种创建交互式、快速动态网页应用的网页开发技术）、
            ASP(是一种创建动态交互式网页并建立强大的 web 应用程序)、
            JSP(是 Java 语言创建动态网页的技术标准) 等技术，它不需要重新加载整个页面内容，就可以实现网页的局部更新。
    httpbin.org 这个网站能测试 HTTP 请求和响应的各种信息，比如 cookie、IP、headers 和登录验证等，
                且支持 GET、POST 等多种方法，对 Web 开发和测试很有帮助。
    爬虫的基本流程:
        发送请求-->获取响应内容-->解析内容-->保存数据
        1.发送请求
            使用http库向目标站点发起请求,即发送一个Request
            Requests包含:请求头,请求体等
        2.获取响应内容
            服务器正常响应,则会得到一个Response
            Response包含:html,json,图片视频等
        3.解析内容
            解析html数据:正则表达式(Re模块),第三方解析库Beautifulsoup等等
            解析json数据:json模块
            解析二进制数据:以wb的方式写进文件
        4.保存数据
            数据库(Mysql,Mongdb,Redis)
            文件
        常见方法:
        urllib.request.urlopen(url,timeout):
            表示向网站发送请求并获取响应对象
            url：表示要爬取数据的 url 地址。
            timeout：设置等待超时时间，指定时间内未得到响应则抛出超时异常。
        urllib.request.Request(url,headers)
            该方法用于创建请求对象、包装请求头，比如重构 User-Agent（即用户代理，指用户使用的浏览器）使程序更像人类的请求，而非机器。
            重构 User-Agent 是爬虫和反爬虫斗争的第一步。
            url：请求的URL地址。
            headers：重构请求头
"""
from urllib.request import urlopen      # 打开网页
from bs4 import BeautifulSoup   # 解析网页内容方式2
import re       # 解析网页内容方式1
from urllib import parse  # 对url进行编码与解码操作
