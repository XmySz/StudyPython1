import dzl
import cv2 as cv
import torchvision.transforms as transforms
import glob
import os
import numpy as np
from PIL import Image
import torch.utils.data as data
import torch
import torch.nn as nn
from utils import load_state_dict_from_url

transf = transforms.Compose(  # 将多个transform组合起来使用
    [
        transforms.ToTensor(),  # 此transform会自行修改正则化的范围为均值0，方差1#
    ]
)


class Data(data.Dataset):
    def __init__(self, root, transforms=None):
        # 所有图片的绝对路径
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, k) for k in imgs]
        self.labels = [int(path.split("-")[0]) for path in imgs]
        # print(self.labels)
        # print(set(self.labels))
        # print(len(set(self.labels)))
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path)
        label = self.labels[index]
        if self.transforms:
            img = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img)
            img = torch.from_numpy(pil_img)

        return img, label

    def __len__(self):
        return len(self.imgs)


batch_size = 64
train_dataSet = Data("/home/sci/zyn/WSI-label/train", transf)
test_dataSet = Data("/home/sci/zyn/WSI-label/test")
train_iter = data.DataLoader(train_dataSet, batch_size=batch_size, shuffle=True)
test_iter = data.DataLoader(test_dataSet, batch_size=batch_size, shuffle=False)
for X,y in train_iter:
    print("X:", X)
    print("y:", y)