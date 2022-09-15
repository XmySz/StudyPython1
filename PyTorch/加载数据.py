"""
    一.dataset(提供一种获取数据的方式及其label值)
    二.dataloader(为网络提供不同的数据形式)
"""

from torch.utils.data import Dataset
from PIL import Image
import os


class MyDate(Dataset):
    def __init__(self, root_dir, label_dir):    # 分别是根路径和标签
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = "E:\\pythonProject\\hymenoptera_data\\train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyDate(root_dir, ants_label_dir)  # 蚂蚁训练集
bees_dataset = MyDate(root_dir, bees_label_dir) # 蜜蜂训练集
train_dataset = ants_dataset + bees_dataset

print(len(train_dataset))