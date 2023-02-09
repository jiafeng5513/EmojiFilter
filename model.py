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
max_epoch = 50                # loop training set by max_epoch times
batch_size = 2                # mini batch size
val_each_iter = 10                 # val on each val_iter mini batch
resize_w = 512                # resize w result in preprocess
resize_h = 512                # resize h result in preprocess


# dataloader
class dataloader(dataset.Dataset):
    def __init__(self, json_path, train=True):
        super(dataloader, self).__init__()
        self.dataset_root, self.dataset_len, self.images = self.read_dataset_json(json_path)
        self.train = train

        # train预处理
        self.train_transforms = transforms.Compose([
            transforms.Resize([resize_h, resize_w]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # test预处理
        self.test_transforms = transforms.Compose([
            transforms.Resize([resize_h, resize_w]),
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
    '''
    self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = norm_layer(self.inplanes)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512 * block.expansion, num_classes)
    '''
    freezd_layer = [model.conv1, model.layer1, model.layer2, model.layer3, model.layer3, model.layer4]
    no_freeze_layer = [model.fc]
    optim_param = []

    for layer in freezd_layer:
        for p in layer.parameters():
            p.requires_grad = False
    for layer in no_freeze_layer:
        for p in layer.parameters():
            p.requires_grad = True
            optim_param.append(p)

    optimizer = torch.optim.SGD(optim_param, lr=learning_rate)
    return model, optimizer


def loss_function(logits, target):
    return F.cross_entropy(logits, target)


def train_and_val(train_set_json_path, val_set_json_path):
    train_dataset = dataloader(json_path=train_set_json_path, train=True)
    val_dataset = dataloader(json_path=val_set_json_path, train=False)

    train_loader = torch.utils.data.dataloader.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = torch.utils.data.dataloader.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    train_mini_batch_number = len(train_loader)
    val_mini_batch_number = len(val_loader)

    model, optimizer = get_model()
    model.train()
    model.to(device)

    iter_total = 1
    for epoch in range(max_epoch):
        for iter, (image, label) in enumerate(train_loader):
            logits = model(image)
            loss = loss_function(logits, label)
            print("epoch {}/{}, iter {}/{}, loss = {}".format(epoch+1, max_epoch,
                                                              iter+1, train_mini_batch_number, loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter_total % val_each_iter == 0:
                print("its time to val")
                for val_iter, (val_image, val_label) in enumerate(val_loader):
                    logits = model(val_image)
                    # 输入一个batch，得到一个batch的输出


            iter_total += 1
    pass


if __name__ == "__main__":
    # train_and_val('./data/dataset.json', './data/dataset.json')

    logits = torch.rand(batch_size, 4)
    print(logits)

    # with torch.no_grad():

