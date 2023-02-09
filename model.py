import os
import time
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.models as models
from torch.utils.data import dataset
import json
import torch.nn.functional as F

# global def
device = torch.device("cpu")  # device
learning_rate = 1e-2          # init learning rate
max_epoch = 6                 # loop training set by max_epoch times
batch_size = 1                # mini batch size
val_iter = 200                # val on each val_iter mini batch


# dataloader
class dataloader(dataset.Dataset):
    def __init__(self, json_path, train=True):
        super(dataloader, self).__init__()
        self.dataset_root, self.dataset_len, self.images = self.read_dataset_json(json_path)
        self.train = train

        # train预处理
        self.train_transforms = transforms.Compose([
            # transforms.Resize(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # test预处理
        self.test_transforms = transforms.Compose([
            # transforms.Resize(20),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        pass

    def read_dataset_json(self, json_path):
        with open(json_path, "r") as json_file:
            dataset = json.load(json_file)
        dataset_root = dataset['root']
        dataset_len = dataset['len']
        images = dataset['images']
        return dataset_root, dataset_len, images

    def __getitem__(self, index):
        image_item = self.images[index]
        item_path = image_item['path']
        item_label = image_item['label']

        img = Image.open(item_path)
        label = int(item_label)

        if self.train:
            img = self.train_transforms(img)
        else:
            img = self.test_transforms(img)

        return img, label

    def __len__(self):
        return self.dataset_len


def get_model(m_path=None):
    model = models.resnet18(pretrained=True)
    # 修改全连接层的输出
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)

    # 加载模型参数
    if m_path is not None:
        checkpoint = torch.load(m_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    return model


def loss_function(logits, target):
    return F.cross_entropy(logits, target)


def train_and_val(train_set_json_path, val_set_json_path):
    train_dataset = dataloader(json_path=train_set_json_path, train=True)
    mini_batch_number = train_dataset.dataset_len // batch_size
    # val_dataset = dataloader(json_path=val_set_json_path, train=False)

    train_loader = torch.utils.data.dataloader.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    # val_loader = torch.utils.data.dataloader.DataLoader(
    #     dataset=val_dataset,
    #     batch_size=batch_size,
    #     shuffle=False
    # )

    model = get_model()
    model.train()
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(max_epoch):
        for iter, (image, label) in enumerate(train_loader):
            logits = model(image)
            loss = loss_function(logits, label)
            print("epoch {}/{}, iter {}/{}, loss = {}".format(epoch+1, max_epoch, iter+1, mini_batch_number, loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    pass


if __name__ == "__main__":
    train_and_val('D:/WorkSpace/jiafeng_projects/EmojiFilter/data/dataset.json', None)

    # with torch.no_grad():

