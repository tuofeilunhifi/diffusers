from PIL import Image
import json
import os
from enchant.checker import SpellChecker

def verify_image(file_path):
    try:
        img = Image.open(file_path)
        img.getdata()
    except OSError:
        return False
    except Image.DecompressionBombError:
        return False
    return True

max_error_count = 100 # 允许的最大错误数
min_text_length = 2 # 允许的最小单词数
def is_in_english(quote): # 定义一个函数来判断句子是否为英文
    d = SpellChecker("en_US") # 创建一个美式英语的拼写检查器
    d.set_text(quote) # 设置要检查的文本
    errors = [err.word for err in d] # 获取所有拼写错误的单词
    return False if ((len(errors) > max_error_count) or len(quote.split()) < min_text_length) else True # 如果错误数超过阈值或单词数太少，则返回False，否则返回True

base_path = "/mnt/workspace/workgroup/yunji.cjy/projects/reptile/wallhaven/train/"
query_list = ['cyberpunk', 'science%20fantasy', 'surreal', 'nebula']
query_list = ['movie']

for query in query_list:
    count = 0
    # 从JSON文件中读取数据
    with open(base_path + "{}_metadata.jsonl".format(query)) as f:
        # 遍历每一行
        err_img_list = []
        err_text_list = []
        for line in f:
            count += 1
            # 去掉换行符
            data = json.loads(line.strip())
            # 打印每一行
            img_path = base_path + data['file_name']
            check_img = verify_image(img_path)
            check_text = is_in_english(data['text'])
            if not check_img:
                print("count:{}, check_img:{}, check_text:{}".format(count, check_img, check_text))
                print(data)
                os.remove(img_path)
                err_img_list.append(data['file_name'])
            if not check_text:
                print("count:{}, check_img:{}, check_text:{}".format(count, check_img, check_text))
                print(data)
                err_text_list.append(data['text'])
        print("err_img_list:{}".format(err_img_list))
        print("err_text_list:{}".format(err_text_list))