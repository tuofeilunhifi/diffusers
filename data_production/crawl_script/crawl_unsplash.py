# 导入需要的库
import threading
import requests
import json
import os
import time

# 定义要爬取的图片数量和关键词
interval_pages = [1, 1000]
per_page = 10
query = "animals" # 关键词

# 创建保存图片的文件夹，如果不存在则创建，存在则跳过
path = "./unsplash/train/" + query + "/"
if not os.path.exists(path):
    os.makedirs(path)
    
# 定义请求的url，使用unsplash的api接口
base_url = 'https://unsplash.com/napi/topics/{}/photos?'.format(query)

# 定义请求头，模拟浏览器访问
headers = {
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
    "Host": "unsplash.com",
    "Referer": "https://unsplash.com/t/" + query,
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
}

def fetch(item, count):
    # 获取图片的id、下载链接和标签列表（alt_description字段）
    img_id = item["id"]
    img_url = item["links"]["download"]
    img_tags = item["alt_description"]

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

    # 增加已爬取的图片数量 
    count += 1 

    # 设置延时，避免过快访问导致被封禁 ip  
    time.sleep(1)
    
count = 0
# 循环发送请求，直到达到指定的图片数量或者没有更多数据为止
for page in range(interval_pages[0], interval_pages[1]):
    url = base_url + "page={}&per_page={}".format(page, per_page)
    print(page, count, url)
    # 发送get请求，并获取响应内容
    response = requests.get(url, headers=headers)

    # 判断响应状态码是否为200，如果不是则跳出循环
    if response.status_code != 200:
        print("请求失败，状态码为", response.status_code)
        break

    # 解析响应内容为json格式，并获取results列表中的数据
    data = response.json()

    # 判断results列表是否为空，如果为空则跳出循环 
    if not data:
     print("没有更多数据了")
     break

    t_list = []
    # 遍历results列表中的每个元素（每个元素代表一张图片）
    for item in data:
        t = threading.Thread(target=fetch, args=(item, count))
        t_list.append(t)
        t.start()

    for t in t_list:
        t.join()
        
    count += per_page