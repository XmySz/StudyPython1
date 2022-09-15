import requests as req
from bs4 import BeautifulSoup

url = input("enter your url: ")

if "http" in url or "https" in url:
    data = req.get(url)
else:
    data = req.get("https://"+url)

soup = BeautifulSoup(data.text, "html.parser")
links = []

for link in soup.find_all("a"):
    links.append(link.get("href"))

with open("../data/mylinks.txt", 'a') as f:
    for i in range(len(links)):
        print(links[i], file=f)   # 注意此处print函数的用法,file参数指定输出位置,默认为sys.stdout