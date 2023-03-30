# https://github.com/mlfoundations/open_clip

import torch
from PIL import Image
import open_clip
import shutil
import json

import os
import torch.nn.functional as F

base_path = "/home/ecs-user/dataset/pinterest/train/"
dst_folder = "/home/ecs-user/dataset/pinterest_filter"
txt_path = "./ori_metadata.jsonl"
new_jsonl = open('metadata.jsonl', 'w')

# 过滤不匹配的图文对
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-H-14')

coca_model, _, transform = open_clip.create_model_and_transforms(
  model_name="coca_ViT-L-14",
  pretrained="mscoco_finetuned_laion2B-s13B-b90k"
)

def clip_similarity(clip_model, image, text):
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = F.cosine_similarity(image_features, text_features, dim=-1)
    return similarity

with open(txt_path) as f:
    count = 0
    # 遍历每一行
    for line in f:
        count += 1
        # 去掉换行符
        data = json.loads(line.strip())
        # 打印每一行
        file_name = '/'.join(data['file_name'].split('/')[-2:])
        img_path = base_path + file_name
        img_text = data['text']

        if os.path.exists(img_path):
            image = Image.open(img_path)
            image = preprocess(image).unsqueeze(0)

            text = tokenizer([img_text])
            similarity = clip_similarity(clip_model, image, text)
            if similarity < 0.3:
                with torch.no_grad(), torch.cuda.amp.autocast():
                    generated = coca_model.generate(image)
                    coca_text = open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")
                    text = tokenizer([coca_text])
                    coca_similarity = clip_similarity(clip_model, image, text)
                    if coca_similarity > similarity:
                        img_text = coca_text
                        similarity = coca_similarity
            
            print("{}, text:{}, similarity:{}".format(file_name, img_text, similarity))
            data = {"file_name": file_name, "text": img_text}
            # 将字典转换为json字符串，并添加换行符
            json_str = json.dumps(data) + "\n"
            # 写入文件
            new_jsonl.write(json_str)
print("filter image-text succssed!")