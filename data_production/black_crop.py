'''
reference: https://blog.csdn.net/weixin_30339457/article/details/99006688
'''
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

src_folder = "/home/ecs-user/dataset/artstation/train"
dst_folder = "/home/ecs-user/dataset/artstation2/train"

for root, dirs, files in os.walk(src_folder):
    for file in files:
        src_file = os.path.join(root, file)
        dst_file = os.path.join(dst_folder, os.path.relpath(src_file, src_folder))
        if not os.path.exists(os.path.dirname(dst_file)):
            os.makedirs(os.path.dirname(dst_file))

        threshold = 14 # 阈值
        try:
            image = cv2.imread(src_file) # 导入图片
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # 转换为灰度图像
        except:
            continue

        nrow = gray.shape[0] # 获取图片尺寸
        ncol = gray.shape[1]

        print(gray[0, 0], gray[100, 75], gray[200, 150], gray[400, 300], gray[800, 600])

        rowc = gray[:,int(1/2*nrow)] # 无法区分黑色区域超过一半的情况
        colc = gray[int(1/2*ncol),:]

        rowflag = np.argwhere(rowc != threshold)
        colflag = np.argwhere(colc != threshold)
        if len(rowflag) == 0 or len(colflag) == 0:
            print(src_file, dst_file)
            continue

        left,bottom,right,top = rowflag[0,0],colflag[-1,0],rowflag[-1,0],colflag[0,0]

        cv2.imwrite(dst_file, image[left:right,top:bottom])
