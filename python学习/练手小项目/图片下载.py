from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import requests as rq
import os
from bs4 import BeautifulSoup
import time

url = "https://wallpapers.com/hd"

output = "output"


def get_url(url):
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get(url)
    print("loading.....")
    res = driver.execute_script("return document.documentElement.outerHTML")

    return res


def get_img_links(res):
    soup = BeautifulSoup(res, 'lxml')
    imglinks = soup.find_all('img', src=True)
    return imglinks


def download_img(img_link, index):
    try:
        extensions = [".jpeg", ".jpg", ".png", ".gif"]
        extension = ".jpg"
        for exe in extensions:
            if img_link.find(exe) > 0:
                extension = exe
                break

        img_data = rq.get(img_link).content
        with open(output + "\\" +str(index + 1) + extension, "wb+") as f:
            f.write(img_data)

        f.close()

    except Exception:
        pass

result = get_url(url)
time.sleep(60)
img_links = get_img_links(result)

if not os.path.isdir(output):
    os.mkdir(output)

for index, img_links in enumerate(img_links):
    img_link = img_links["src"]
    print("Downloading...")
    if img_link:
        download_img(img_link, index)
print("Download Complete!!")
