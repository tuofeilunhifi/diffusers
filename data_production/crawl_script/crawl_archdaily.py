# 导入需要的库
import os
import time
import json
import re
from bs4 import BeautifulSoup
import threading
import requests

from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from selenium.common.exceptions import TimeoutException

def retry_requests_get(url, retries=3):
    session = requests.Session()
    retry = Retry(total=retries, backoff_factor=0.1)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    return session.get(url).content

def retry_get(driver, url, retries=3):
    for i in range(retries):
        try:
            driver.get(url)
            return True
        except TimeoutException:
            print(f"Timeout exception occurred. Retrying {i+1} of {retries}...")
    return False

def fetch(driver, url, path, query):
    print(url)
    # 发送get请求，并获取响应内容
    retry_get(driver, url, retries=100)

    # 获取图片的id、下载链接和标签列表（alt_description字段）
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    
    img_info = soup.find('div', id='gallery-items').find_all('img')

    url_dict = {}
    for i in range(len(img_info)):
        if i % 1 == 0:
            url_dict[img_info[i].get("id")] = '/'.join(url.split('/')[:-1]) + '/' + img_info[i].get("id")
    print(len(url_dict))

    img_dict = {}
    for img_id, url in url_dict.items():
        # 发送get请求，并获取响应内容
        retry_get(driver, url, retries=100)

        # 获取图片的id、下载链接和标签列表（alt_description字段）
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        
        img_url = soup.find('div', id='gallery-items').find('img', id=img_id).get('src')
        img_title = soup.find('div', class_='afd-gal-img-meta__title-container').find('h1').text
        img_type = soup.find('div', class_='afd-gal-img-meta__title-container').find('p').text
        img_tag = soup.find('div', class_='afd-gal__tags js-gal-tags').find_all('a')
        img_text = img_title + ',' + img_type + ',' + ','.join([tag.text for tag in img_tag]) + '.'
        print(img_url, img_text)

        # 拼接图片保存的路径和文件名（以id命名）
        img_path = path + "{}.jpg".format(img_id)

        # 发送get请求，并获取图片二进制数据
        img_data = retry_requests_get(img_url, retries=100)

        # 打开文件，并以二进制写入模式写入图片数据 
        if not os.path.exists(img_path): # 返回True或False
            with open(img_path,"wb") as f: 
                f.write(img_data) 

        # 打开文件，并以追加模式写入标签数据（每行一个字典）
        with open(path + "{}_metadata.jsonl".format(query), "a") as f:
            # 创建一个字典，包含文件名和附加特征
            data = {"file_name": img_path, "text": img_text}
            # 将字典转换为json字符串，并添加换行符
            json_str = json.dumps(data) + "\n"
            # 写入文件
            f.write(json_str)

interval_pages = [2, 1000]
base_url = 'https://www.archdaily.com/search/images?page={}'

for page in range(interval_pages[0], interval_pages[1]):
    # 创建选项对象并添加"--headless"参数
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)

    url = base_url.format(page)
    # 進入故宮網頁
    retry_get(driver, url, retries=100)

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    a_tags = soup.find_all('div', class_='project-object__save-buttons')
    tags_list = []
    for tag in a_tags:
        if tag.select_one('a') and tag.select_one('a')['href']:
            tags_list.append(tag.select_one('a')['href'])

    print(tags_list, url, page)

    # 如果資料夾不存在就新增
    img_dir = "./archdaily/train/" + str(page) + "/"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    for url in tags_list:
        if os.path.exists(img_dir + "{}_metadata.jsonl".format(page)):
            f_read = open(img_dir + "{}_metadata.jsonl".format(page), 'r').read()
            img_id = url.split('/')[-1]
            if img_id in f_read:
                continue
        fetch(driver, url, img_dir, page)
