# https://github.com/LAION-AI/LAION-5B-WatermarkDetection
# https://github.com/LAION-AI/CLIP-based-NSFW-Detector
# https://github.com/LAION-AI/aesthetic-predictor
# https://github.com/rom1504/clip-retrieval/blob/main/clip_retrieval/h14_nsfw_model.py

import os
import shutil
import timm
import torch.nn.functional as F

from PIL import Image
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
import clip
import autokeras as ak  # pylint: disable=import-outside-toplevel
from tensorflow.keras.models import load_model  # pylint: disable=import-outside-toplevel

# if you changed the MLP architecture during training, change it also here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

def is_image(filename):
    return os.path.splitext(filename)[1].lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

def filter_images(txt_path, src_folder, dst_folder, preprocess, aesthetic_backbone, aesthetic_mlp, pwatermark_model, safety_model):
    with open(txt_path, 'r') as readfile:
        txt_content = readfile.read()
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_folder, os.path.relpath(src_file, src_folder))
            if not os.path.exists(os.path.dirname(dst_file)):
                os.makedirs(os.path.dirname(dst_file))

            filter_flag = False
            if is_image(src_file):
                if file not in txt_content:
                    print("file not in txt!")
                    filter_flag = True
                
                if not filter_flag:
                    img = Image.open(src_file)
                    width, height = img.size

                    # 按照图片长宽过滤
                    if width < 512 or height < 512 or width / height >= 3 or height / width >= 3:
                        print("width:{}, height:{}".format(width, height))
                        filter_flag = True

                    # 按照美学/水印/色情过滤
                    if not filter_flag:
                        image = preprocess(img).unsqueeze(0).to(device)
                        # 按照美学过滤
                        with torch.no_grad():
                            image_features = aesthetic_backbone.encode_image(image)
                        im_emb_arr = normalized(image_features.cpu().detach())
                        aesthetic_score = aesthetic_mlp(im_emb_arr.to(device).type(torch.cuda.FloatTensor))
                        if aesthetic_score < 4:
                            print("aesthetic_score:{}".format(aesthetic_score))
                            filter_flag = True
                        # 按照水印过滤
                        if not filter_flag:
                            with torch.no_grad():
                                pred = pwatermark_model(image)
                                pwatermark_score = F.softmax(pred, dim=1).detach().cpu().numpy().tolist()[0]
                                if pwatermark_score[0] > 0.9:
                                    print("pwatermark_score:{}".format(pwatermark_score[0]))
                                    filter_flag = True
                        # 按照色情过滤
                        if not filter_flag:
                            with torch.no_grad():
                                image_features = aesthetic_backbone.encode_image(image)
                            emb = np.asarray(normalized(image_features.cpu().detach()))
                            nsfw_score = safety_model.predict(emb)
                            if nsfw_score > 0.98:
                                print("nsfw_score:{}".format(nsfw_score))
                                filter_flag = True

            if filter_flag:
                print(src_file)
                shutil.move(src_file, dst_file)

src_folder = "/home/ecs-user/dataset/pinterest/train"
dst_folder = "/home/ecs-user/dataset/pinterest_filter"
txt_path = "./ori_metadata.jsonl"

# 合并txt文件
jsonl_all = []
with open(txt_path, 'w') as outfile:
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            if file.endswith('.jsonl'):
                with open(root + '/' + file, 'r') as readfile:
                    outfile.write(readfile.read())
print("merge succssed!")

# 数据去重
with open(txt_path) as f:
    lines = f.readlines()
lines = list(set(lines))
with open(txt_path, 'w') as f:
    f.writelines(lines)
print("remove succssed!")

aesthetic_model_path = "./model_dir/sac+logos+ava1-l14-linearMSE.pth"
pwatermark_model_path = './model_dir/watermark_model_v1.pt'
safety_model_path = "./model_dir/clip_autokeras_binary_nsfw"

device = "cuda" if torch.cuda.is_available() else "cpu"
aesthetic_backbone, preprocess = clip.load("ViT-L/14", device=device)
aesthetic_mlp = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
s = torch.load(aesthetic_model_path)   # load the model you trained previously or the model available in this repo
aesthetic_mlp.load_state_dict(s)
aesthetic_mlp.to("cuda")
aesthetic_mlp.eval()
print("end load aesthetic model!")

pwatermark_model = timm.create_model(
    'efficientnet_b3a', pretrained=True, num_classes=2)
pwatermark_model.classifier = nn.Sequential(
    # 1536 is the orginal in_features
    nn.Linear(in_features=1536, out_features=625),
    nn.ReLU(),  # ReLu to be the activation function
    nn.Dropout(p=0.3),
    nn.Linear(in_features=625, out_features=256),
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=2),
)
pwatermark_model.load_state_dict(torch.load(pwatermark_model_path))
pwatermark_model.eval()
if torch.cuda.is_available():
    pwatermark_model.cuda()
print("end load pwatermark model!")

safety_model = load_model(safety_model_path, custom_objects=ak.CUSTOM_OBJECTS)
print("end load safety model!")

filter_images(txt_path, src_folder, dst_folder, preprocess=preprocess, aesthetic_backbone=aesthetic_backbone, aesthetic_mlp=aesthetic_mlp, pwatermark_model=pwatermark_model, safety_model=safety_model)