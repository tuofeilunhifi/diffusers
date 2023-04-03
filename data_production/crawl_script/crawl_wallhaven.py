# 导入需要的库
import threading
import requests
import json
import os
import time
from urllib3.util import Retry
from requests.adapters import HTTPAdapter
from bs4 import BeautifulSoup

def fetch_img_info(url, count, path, query):
    # 发送get请求，并获取响应内容
    response = session.get(url, headers=headers)

    # 判断响应状态码是否为200，如果不是则跳出循环
    if response.status_code != 200:
        print("请求失败，状态码为", response.status_code)
        return 

    # 获取图片的id、下载链接和标签列表（alt_description字段）
    soup = BeautifulSoup(response.text, 'html.parser')
    img_info = soup.find('img', id='wallpaper')
    img_url = img_info.get("src")
    img_tags = img_info.get("alt")
    img_id = os.path.basename(url)

    if img_id is None or img_url is None or img_tags is None:
        return 

    # 打印图片信息
    print("正在下载第", count+1, "张图片：")
    print("id:", img_id)
    print("url:", img_url)
    print("tags:", img_tags)

    # 拼接图片保存的路径和文件名（以id命名）
    img_path = path + img_id + ".jpg"

    # 发送get请求，并获取图片二进制数据 
    img_data = requests.get(img_url).content 

    # 打开文件，并以二进制写入模式写入图片数据 
    if not os.path.exists(img_path): # 返回True或False
        with open(img_path,"wb") as f: 
          f.write(img_data) 

    # 打开文件，并以追加模式写入标签数据（每行一个字典）
    with open(query + "_metadata.jsonl","a") as f:
        # 创建一个字典，包含文件名和附加特征
        data = {"file_name": img_path, "text": img_tags}
        # 将字典转换为json字符串，并添加换行符
        json_str = json.dumps(data) + "\n"
        # 写入文件
        f.write(json_str)

    # 设置延时，避免过快访问导致被封禁 ip  
    time.sleep(1)
    
def fetch_pages(session, interval_pages, base_url, per_page, path, query):
    count = 0
    # 循环发送请求，直到达到指定的图片数量或者没有更多数据为止
    for page in range(interval_pages[0], interval_pages[1]):
        url = base_url + "page={}".format(page)
        print(page, count, url)
        # 发送get请求，并获取响应内容
        response = session.get(url, headers=headers)

        # 判断响应状态码是否为200，如果不是则跳出循环
        if response.status_code != 200:
            print("请求失败，状态码为", response.status_code)
            break

        # 解析响应内容为json格式，并获取results列表中的数据
        soup = BeautifulSoup(response.text, 'html.parser')
        a_tags = soup.find_all('a', class_='preview')
        data = [tag.get("href") for tag in a_tags] #获取每个 a 标签的 href 属性的值，并存入一个列表

        # 判断results列表是否为空，如果为空则跳出循环 
        if not data:
         print("没有更多数据了")
         break

        t_list = []
        # 遍历results列表中的每个元素（每个元素代表一张图片）
        for url in data:
            t = threading.Thread(target=fetch_img_info, args=(url, count, path, query))
            t_list.append(t)
            t.start()

        for t in t_list:
            t.join()

        count += per_page
    return count
    
if __name__ == '__main__':
    # 定义请求头，模拟浏览器访问
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Connection": "keep-alive",
        "Host": "wallhaven.cc",
        "Referer": "https://wallhaven.cc/",
        "Cookie": "_pk_id.1.01b8=ae2017656e09c3f3.1678788093.; _pk_ref.1.01b8=%5B%22%22%2C%22%22%2C1679383684%2C%22https%3A%2F%2Faliyuque.antfin.com%2F%22%5D; _pk_ses.1.01b8=1; XSRF-TOKEN=eyJpdiI6ImVnbUNWTXdxVzRxOWcwTlVIK1d3K2c9PSIsInZhbHVlIjoicHBlWTRCemlYblBoVjkxbDVTUE0zcGQ0bVJQaFRSUnRnVWRqNTUxb0dPMDd3Z1huNTBEZEExUE5pSVRsYlZtQiIsIm1hYyI6IjJkNzdjNjI5ZDJjYzAyYjNkOGU2Y2IyNjZkMGQ2ZTk1MDI0YjVhY2VmNDQ5N2NlMGNhNDE5NGEwN2I5OGI4MDEifQ%3D%3D; wallhaven_session=eyJpdiI6InV0QWhwUG1mNVRlT3R5SG0wK3Jaamc9PSIsInZhbHVlIjoiS0l4NEVpdVBvVWI2ZDduTmJZODc2SXVDMkxMRW40cHFxb2cxTFZuSWRLc1pPMHZpbWhrTHJZaHNJZW1mV0F2VyIsIm1hYyI6IjQ5M2E1M2ZiZDc2ZGY2MmJiNWE3ZjNiYTk5N2VjMzRlOTI1OGE3OGQ0MGU2NjMyZmZlZjU2NzgwMmU0NzFlM2IifQ%3D%3D",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"
    }
    
    session = requests.Session()
    retries = Retry(total=10, # 最大重试次数
                    backoff_factor=1, # 重试间隔因子
                    status_forcelist=[500]) # 需要重试的状态码列表
    session.mount("https://", HTTPAdapter(max_retries=retries)) # 将重试策略应用到会话对象上
    
    # 定义要爬取的图片数量和关键词
    # query_list = ['science%20fiction', 'future']
    # query_list = ['artwork']
    query_list = ['cyberpunk', 'science%20fantasy', 'surreal', 'nebula', 'movie']
    interval_pages = [1, 1000]
    per_page = 24
    base_data_path = "./wallhaven/train/"

    for query in query_list:

        # 创建保存图片的文件夹，如果不存在则创建，存在则跳过
        path = base_data_path + query + "/"
        if not os.path.exists(path):
            os.makedirs(path)

        # 定义请求的url，使用unsplash的api接口
        base_url = 'https://wallhaven.cc/search?q={}&categories=110&purity=100&sorting=relevance&order=desc&'.format(query)

        count = fetch_pages(session, interval_pages, base_url, per_page, path, query)
        
        print("query:{}, count:{}".format(query, count))