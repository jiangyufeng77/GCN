from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random

image_dir = '/media/ouc/4T_A/jiang/multi_domain_model/stargan/data/celeba/images'
attr_path = '/media/ouc/4T_A/jiang/multi_domain_model/stargan/data/celeba/list_attr_celeba.txt'
out_dir = '/media/ouc/4T_A/jiang/multi_domain_model/stargan/data/celeba_split/Young'
selected_attrs = ['Young']



# 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
# 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
# 'Wearing_Necktie', 'Young'

os.makedirs(out_dir, exist_ok=True)
images_listdir = os.listdir(image_dir)
attr2idx = {}
idx2attr = {}
train_dataset = []

lines = [line.rstrip() for line in open(attr_path, 'r')]
all_attr_names = lines[1].split()
for i, attr_name in enumerate(all_attr_names):
    attr2idx[attr_name] = i
    idx2attr[i] = attr_name

lines = lines[2:]
# random.seed(1234)
# random.shuffle(lines)
for i, line in enumerate(lines):
    split = line.split()
    filename = split[0]
    values = split[1:]

    label = []
    for attr_name in selected_attrs:
        idx = attr2idx[attr_name]
        label.append(values[idx] == '1')

    train_dataset.append([filename, label])

for i, img in enumerate(train_dataset):
    # print(img[1][0])
    if img[1][0] == True:
        # image_out = Image.open(images_listdir)
        # image_out.save(out_dir + '/' + selected_attrs[0])
        image_out = Image.open(os.path.join(image_dir, img[0]))
        image_out.save(out_dir + '/' + img[0])
        # print(img[0])

