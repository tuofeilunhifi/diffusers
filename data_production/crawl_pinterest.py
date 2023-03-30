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
    # img_info = soup.find('div', attrs={'data-test-id': 'visual-content-container'}).find_all('img')
    # title_info1 = soup.find('div', attrs={'data-test-id': 'closeup-title'})
    # title_info2 = soup.find('div', attrs={'data-test-id': 'product-title'})
    # rich_info1 = soup.find('div', attrs={'data-test-id': 'description'})
    # rich_info2 = soup.find_all('div', attrs={'data-test-id': 'structured-description'})
    
    img_info = soup.find('div', attrs={'data-test-id': 'pin-closeup-image'})
    title_info1 = soup.find('div', attrs={'data-test-id': 'CloseupDetails'})
    # title_info2 = soup.find('div', attrs={'data-test-id': 'product-title'})
    rich_info1 = soup.find('div', attrs={'data-test-id': 'CloseupDescriptionContainer'})
    # rich_info2 = soup.find_all('div', attrs={'data-test-id': 'structured-description'})
    
    if img_info is not None and img_info.find_all('img'):
        img_info = img_info.find_all('img')[-1]
    else:
        return 

    img_url = img_info.get("src")

    describe_list = []

    img_tags = img_info.get("alt")
    if len(img_tags) > 1:
        describe_list.append(img_tags)

    if title_info1 is not None and title_info1.select_one('h1'):
        img_title = title_info1.select_one('h1').text
    # elif title_info2 is not None and title_info2.select_one('h2'):
    #     img_title = title_info2.select_one('h2').text
    else:
        img_title = ''
    if len(img_title) > 0:
        describe_list.append(img_title)
    img_text = '.'.join(describe_list)

    if len(img_text.split()) < 5:
        if rich_info1 is not None and rich_info1.select_one('span'):
            img_rich = rich_info1.text
        # elif rich_info2 is not None:
        #     rich_list = []
        #     for rich in rich_info2:
        #         rich_list.append(rich.text)
        #     img_rich = ''.join(rich_list)
        else:
            img_rich = ''
        if len(img_rich) > 0:
            img_text = img_text + '.' + img_rich

    if query.lower().replace('%20', ' ') not in img_text.lower():
        img_text = query.replace('%20', ' ') + '.' + img_text
    img_id = url.split('/')[-2]

    print(img_id, img_url, img_text)

    if len(img_id)==0 or len(img_url)==0 or len(img_text)==0:
        return 

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

    # 设置延时，避免过快访问导致被封禁 ip  
    # time.sleep(1)

keywords_txt = './v4_midjourney_artist.txt'
with open(keywords_txt) as f:
    lines = f.readlines()
keywords = []
for line in lines:
    line = line.strip()
    if len(line) > 0:
        keywords.append(line)
print(keywords)

base_url = "https://www.pinterest.com/search/pins/?q={}&rs=typed"

# 创建选项对象并添加"--headless"参数
options = webdriver.ChromeOptions()
options.add_argument("--headless")
driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)

# 点击登录按钮
# print("login")
# driver.get("https://www.pinterest.com/search/pins/?q=Esao%20Andrews&rs=typed")
# login_button = driver.find_element_by_xpath('//div[@data-test-id="login-button"]//button[@type="button"]')
# login_button.click()
# # 设置延时，避免过快访问导致被封禁 ip  
# time.sleep(5)

# # 输入用户名和密码
# username = driver.find_element_by_name('id')
# password = driver.find_element_by_name('password')
# username.send_keys('xxx')
# password.send_keys('xxx')

# # 点击登录按钮
# login_button = driver.find_element_by_xpath('//button[@type="submit"]')
# login_button.click()
# # 设置延时，避免过快访问导致被封禁 ip  
# time.sleep(5)
# print("login_end")

count = 0
for keyword in keywords:
    keyword = keyword.replace(' ', '%20')
    url = base_url.format(keyword)

    # 如果資料夾不存在就新增
    img_dir = "./pinterest/train/" + keyword + "/"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    # 進入故宮網頁
    retry_get(driver, url, retries=100)

    SCROLL_PAUSE_TIME = 5
    max_pages = 10

    page = 0
    links = []
    while page < max_pages:
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        a_tags = soup.find_all('div', class_='Yl- MIw Hb7')
        tags_list = []
        for tag in a_tags:
            if tag.select_one('a') and tag.select_one('a')['href']:
                tags_list.append("https://www.pinterest.com" + tag.select_one('a')['href'])
        links.extend(tags_list) #获取每个 a 标签的 href 属性的值，并存入一个列表
        # Scroll down to bottom
        driver.execute_script("window.scrollBy(0, 2 * window.innerHeight);")

        # Wait to load page
        time.sleep(SCROLL_PAUSE_TIME)

        page += 1
        print("keyword:{}, page:{}, max_pages:{}".format(keyword, page, max_pages))

    links = set(links)
    print(len(links))

    # 遍历results列表中的每个元素（每个元素代表一张图片）
    for url in links:
        if os.path.exists(img_dir + "{}_metadata.jsonl".format(keyword)):
            f_read = open(img_dir + "{}_metadata.jsonl".format(keyword), 'r').read()
            img_id = url.split('/')[-2]
            if img_id in f_read:
                continue
        fetch(driver, url, img_dir, keyword)
        
    count += len(links)
