from src.dataset import VOCDetection

test_dataset = VOCDetection(
    path='data/dataset',
    split='Main/test'
)
len(test_dataset)

import random
import os

# read all names
path = 'data/dataset/ImageSets/Main/trainval.txt'
all_names = open(path, 'r').readlines()
n = len(all_names)
print(n)

# define train/val proportions
val_size = len(test_dataset)
train_size = n - val_size
print(train_size, val_size)

# shuffle and divide
random.seed(2)
random.shuffle(all_names)
train_names = all_names[:train_size]
val_names = all_names[:val_size]

# save
my_path = 'data/dataset/ImageSets/my_splits'
if not os.path.exists(my_path):
    os.makedirs(my_path)

train_path = os.path.join(my_path, 'train.txt')
val_path = os.path.join(my_path, 'val.txt')

open(train_path, 'w').writelines(train_names)
open(val_path, 'w').writelines(val_names)