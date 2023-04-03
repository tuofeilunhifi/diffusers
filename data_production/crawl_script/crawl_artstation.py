# 导入需要的库
import os
import time
import json
from json.decoder import JSONDecodeError
import re
from bs4 import BeautifulSoup
import threading
import requests

from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from selenium.common.exceptions import TimeoutException, WebDriverException
from fake_useragent import UserAgent

def retry_requests_get(url, retries=100):
    session = requests.Session()
    retry = Retry(total=retries, backoff_factor=0.1)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    return session.get(url).content

def retry_get(driver, url, retries=100):
    for i in range(retries):
        try:
            driver.get(url)
            return True
        except TimeoutException:
            print(f"Timeout exception occurred. Retrying {i+1} of {retries}...")
        except WebDriverException:
            return False
    return False

def retry_find(soup, retries=100):
    for i in range(retries):
        try:
            img_infos = soup.find('main').find_all('picture')
            return img_infos
        except AttributeError:
            print(f"AttributeError. Retrying {i+1} of {retries}...")
    return False

def retry_save(driver, img_path, retries=100):
    for i in range(retries):
        try:
            driver.save_screenshot(img_path)
            return True
        except TimeoutException:
            print(f"TimeoutException. Retrying {i+1} of {retries}...")
    return False

def fetch(driver, url, path, query):
    # 创建选项对象并添加"--headless"参数

    # 发送get请求，并获取响应内容
    if not retry_get(driver, url, retries=100):
        return 

    time.sleep(10)

    # 获取图片的id、下载链接和标签列表（alt_description字段）
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    
    img_infos = retry_find(soup, retries=100)

    if not img_infos:
        return

    # 遍历每个picture标签，再找到它下面的img标签
    img_url_list = []
    for img_info in img_infos:
        img_url_list.append(img_info.find('img')['src'])

    img_id = url.split('/')[-1]
    # img_author = soup.find('div', class_='project-author').text
    img_title = soup.find('perfect-scrollbar').find('h1').text
    img_tag = soup.find('perfect-scrollbar').find('footer').find_all('ul')[1].find_all('a')
    img_tag = ','.join([tag.text.strip('#') for tag in img_tag])
    img_text = img_title + ',' + img_tag + '.'
    print("url:{}, img_text:{}".format(url, img_text))

    for i, img_url in enumerate(img_url_list):
        print("count:{}, img_url:{}".format(i, img_url))
        # 拼接图片保存的路径和文件名（以id命名）
        img_path = path + "{}.jpg".format(img_id + '_{}'.format(i))

        if not retry_get(driver, img_url, retries=100):
            continue
        
        if not retry_save(driver, img_path, retries=100):
            continue

        # 发送get请求，并获取图片二进制数据
        # img_data = retry_requests_get(img_url, retries=100)
        # print(i, img_url)

        # # 打开文件，并以二进制写入模式写入图片数据 
        # if not os.path.exists(img_path): # 返回True或False
        #     with open(img_path,"wb") as f: 
        #         f.write(img_data) 

        # 打开文件，并以追加模式写入标签数据（每行一个字典）
        with open(path + "{}_metadata.jsonl".format(query), "a") as f:
            # 创建一个字典，包含文件名和附加特征
            data = {"file_name": img_path, "text": img_text}
            # 将字典转换为json字符串，并添加换行符
            json_str = json.dumps(data) + "\n"
            # 写入文件
            f.write(json_str)

# {“解剖学”:69，“摘要”:70，“动物与野生动物”:71，“动画与漫画”:72，
# “建筑可视化”:73，“角色设计”:74，“角色建模”:75，
# “编辑插画”:76，“儿童艺术”:77，“漫画艺术”:78，
# “生物与怪物”:80，“环境概念艺术与设计”:81，“粉丝艺术”:82，
# “幻想”:83，“封面艺术”:84，“游戏艺术”:85，“恐怖”:86，“平面设计”:87，
# “插图”:88，“工业与产品设计”:89，“照明”:90，
# “哑光绘画”:91，“机甲”:92，“像素和体素”:93，“道具”:94，
# “科幻小说”:95，“故事板”:96，“纹理和材料”:97，“教程”:98，
# “用户界面”:99，“车辆”:100，“建筑概念”:101，
# “网络和应用程序设计”:102，“棋盘和纸牌游戏艺术”:103，“书籍插图”:104，
# “角色动画”:105，“时尚与服装设计”:106，“游戏玩法与关卡设计”:107，
# “游戏和实时3D环境艺术”:108，“硬表面”:109，
# “机械设计”:110，“运动图形学”:111，“摄影测量与3d扫描”:112，
# “肖像”:113，“现实主义”:114，“科学插画与可视化”:115，“脚本与工具”:116，
# “素描”:117，“静物”:118，“程式化”:119，“技术艺术”:120，
# “玩具和收藏品”:121，“电影、电视和动画视觉特效”:122，“实时和游戏视觉特效”:123，
# “虚拟和增强现实”:124，“武器”:126，“虚幻引擎”:127，“汽车”:128，
# “现实扫描”:8064，“命运的艺术2光落”:8668，“圣徒与罪人的艺术:惩罚”:8684}

# keywords_dict = {'Anatomy': 69, 'Abstract': 70, ' Animals&Wildlife': 71, ' Anime&Manga': 72, 
# ' Architectural Visualization': 73, 'Character Design': 74, ' Character Modeling': 75, 
# ' Editorial Illustration': 76, " Children's Art": 77, ' Comic Art': 78, 
# ' Creatures&Monsters': 80, ' Environmental Concept Art&Design': 81, ' Fan Art': 82,
# 'Fantasy': 83, 'Cover Art': 84, ' Game Art': 85, 'Horror': 86, ' Graphic Design': 87,
# 'Illustration': 88, ' Industrial&Product Design': 89, 'Lighting': 90,
# ' Matte Painting': 91, 'Mecha': 92, ' Pixel&Voxel': 93, 'Props': 94, 
# ' Science Fiction': 95, 'Storyboards': 96, ' Textures&Materials': 97, 'Tutorials': 98,
# ' User Interface': 99, 'Vehicles': 100, ' Architectural Concepts': 101, 
# ' Web and AppDesign': 102, ' Board and Card Game Art': 103, ' Book Illustration': 104,
# 'Character Animation': 105, ' Fashion&Costume Design': 106, ' Gameplay&Level Design': 107, 
# ' Games and Real-Time 3D Environment Art': 108, ' Hard Surface': 109, 
# ' Mechanical Design': 110, ' Motion Graphics': 111, ' Photogrammetry&3D Scanning': 112, 
# 'Portraits': 113, 'Realism': 114, ' Scientific Illustration&Visualization': 115, ' Scripts&Tools': 116, 
# 'Sketches': 117, ' Still Life': 118, 'Stylized': 119, ' Technical Art': 120,
# ' Toys&Collectibles': 121, ' VFX for Film,TV&Animation ': 122, ' VFX for Real-Time&Games': 123, 
# ' Virtual and Augmented Reality': 124, 'Weapons': 126, 'Unreal Engine': 127, 'Automotive': 128,
# 'RealityScan': 8064, ' The Art of Destiny 2 Lightfall': 8668, 'The Art of Saints & Sinners: Retribution': 8684}

# 1
# keywords_dict = {'Portraits': 113, 'Gameplay&Level Design': 107}
# 'Anatomy': 69, 

# 2
# keywords_dict = {' Character Modeling': 75}
# 'Character Design': 74

# 3
# keywords_dict = {'Mecha': 92, 'Vehicles': 100}

# 4
# keywords_dict = {' Motion Graphics': 111, 'Realism': 114}
# ' Mechanical Design': 110

# 5
# keywords_dict = {' Scripts&Tools': 116}
# ' Scientific Illustration&Visualization': 115

# 6
# keywords_dict = {' VFX for Real-Time&Games': 123, ' Virtual and Augmented Reality': 124}
# ' VFX for Film,TV&Animation ': 122, 

# 7
# keywords_dict = {'Unreal Engine': 127, 'Automotive': 128}
# 'Weapons': 126, 

# 8
# keywords_dict = {' Hard Surface': 109}
# ' Games and Real-Time 3D Environment Art': 108, 

# 9
# keywords_dict = {' Architectural Visualization': 73}
# ' Architectural Concepts': 101, 

# 10
# keywords_dict = {'Character Animation': 105}
# ' Game Art': 85, 

# 11
# keywords_dict = {' Still Life': 118}
# 'Sketches': 117, 

# 12
# keywords_dict = {'Automotive': 128}
# 'Vehicles': 100, 

# 13
# keywords_dict = {' Science Fiction': 95}
# ' Animals&Wildlife': 71, 

# 14
keywords_dict = {'Illustration': 88}
# ' Environmental Concept Art&Design': 81, 

interval_pages = [1, 1000]
base_url = 'https://www.artstation.com/api/v2/community/channels/projects.json?channel_id={}&page={}&sorting=popular&dimension=all&per_page=30'

# 创建选项对象并添加"--headless"参数
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_experimental_option("excludeSwitches", ["enable-automation"])  # 禁止浏览器被监控提示
options.add_experimental_option('detach', True) # 不自动关闭浏览器
options.add_argument('--incognito')  # 无痕模式
options.add_argument('--disable-gpu') # 禁用gpu加速
options.add_argument('--hide-scrollbars')  # 隐藏滚动条
options.add_argument("--user-agent={}".format(UserAgent().random))  # 设置请求头user-agent
driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)

# retry_get(driver, "https://www.artstation.com", retries=100)
# html = driver.page_source
# soup = BeautifulSoup(html, 'html.parser')
# keywords = soup.find('ngx-slick-carousel').find_all('a')
# keywords_list = []
# for keyword in keywords:
#     keywords_list.append(keyword.text)
# print(keywords_list)

for keyword, id in keywords_dict.items():
    keyword = keyword.strip().replace(' ', '%20')
    for page in range(interval_pages[0], interval_pages[1]):
        options.add_argument("--user-agent={}".format(UserAgent().random))  # 设置请求头user-agent
        driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)

        url = base_url.format(id, page)
        # 進入故宮網頁
        if not retry_get(driver, url, retries=100):
            continue 

        time.sleep(1)

        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        try:
            json_soup = json.loads(soup.text)
        except JSONDecodeError:
            continue
        tags_list = []
        for item in json_soup['data']:
            tags_list.append(item['url'])
        print(tags_list)
        print("url_num:{}, id:{}, page:{}".format(len(tags_list), id , page))

        # 如果資料夾不存在就新增
        img_dir = "./artstation/train/" + str(keyword) + "/"
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        for url in tags_list:
            if os.path.exists(img_dir + "{}_metadata.jsonl".format(keyword)):
                f_read = open(img_dir + "{}_metadata.jsonl".format(keyword), 'r').read()
                img_id = url.split('/')[-1]
                if img_id in f_read:
                    continue
            fetch(driver, url, img_dir, keyword)
        driver.quit()