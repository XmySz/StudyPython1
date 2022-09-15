"""
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
            接信息json数据:json模块
            解析二进制数据:以wb的方式写进文件
        4.保存数据
            数据库(Mysql,Mongdb,Redis)
            文件
        C:/Users/zhaolg/AppData/Local/Temp/20170427113109368.png
"""
from urllib.request import urlopen      # 打开网页
from bs4 import BeautifulSoup   # 解析网页内容2
import re       # 解析网页内容方式1

html = urlopen(
        "https://mofanpy.com/static/scraping/list.html"
).read().decode('utf-8')    # decode()转化中文显示

print(html)