import json
import os.path

import torch.nn as nn
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import dataset
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from GlobalConsts import *
from utils import *

# global def
device = torch.device("cuda")  # device


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
        if not os.path.exists(json_path) or not os.path.isfile(json_path):
            raise RuntimeError("{} is not exists or it is not a regular file!".format(json_path))
        dataset_root = os.path.dirname(os.path.abspath(json_path))
        with open(json_path, "r") as json_file:
            dataset = json.load(json_file)
        dataset_root = dataset['root']
        dataset_len = dataset['len']
        images = dataset['images']
        return dataset_root, dataset_len, images

    def __getitem__(self, index):
        if index not in range(len(self.images)):
            raise RuntimeError("image index out of range!")
        image_item = self.images[index]

        item_path = os.path.join(self.dataset_root, image_item['path'].lstrip('/'))
        if not os.path.exists(item_path) or not os.path.isfile(item_path):
            raise RuntimeError("{} is not exists or it is not a regular file!".format(item_path))

        item_label = image_item['label']
        if item_label not in range(CLASSES_COUNT):
            raise RuntimeError("label = {} of {} is out of range!".format(item_label, item_path))

        img = Image.open(item_path)
        img = img.convert("RGB")

        label = int(item_label)

        if self.train:
            img = self.train_transforms(img)
        else:
            img = self.test_transforms(img)

        return img, label

    def __len__(self):
        return self.dataset_len


class classifer(nn.Module):
    def __init__(self, in_ch, num_classes):
        super(classifer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # import pdb;pdb.set_trace()
        return x


class EmojiNet(nn.Module):
    def __init__(self, num_class, pretrained=True):
        super(EmojiNet, self).__init__()
        model = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(model.children())[:-3])  # 把最后的layer4,Avgpool和Fully Connected Layer去除
        self.classification_head1 = nn.Sequential(*list(model.children())[-3], classifer(2048, num_class))

    def forward(self, x):
        x = self.backbone(x)
        output = self.classification_head1(x)
        return output


def get_model(m_path=None):
    model = EmojiNet(pretrained=True, num_class=CLASSES_COUNT)
    # # 修改全连接层的输出
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, CLASSES_COUNT)

    # # 加载模型参数
    # if m_path is not None:
    #     checkpoint = torch.load(m_path)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    # '''
    # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
    # self.bn1 = norm_layer(self.inplanes)
    # self.relu = nn.ReLU(inplace=True)
    # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    # self.layer1 = self._make_layer(block, 64, layers[0])
    # self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
    # self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
    # self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
    # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    # self.fc = nn.Linear(512 * block.expansion, num_classes)
    # '''
    # freezd_layer = [model.conv1, model.layer1, model.layer2, model.layer3, model.layer3, model.layer4]
    # no_freeze_layer = [model.fc]
    # optim_param = []
    #
    # for layer in freezd_layer:
    #     for p in layer.parameters():
    #         p.requires_grad = False
    # for layer in no_freeze_layer:
    #     for p in layer.parameters():
    #         p.requires_grad = True
    #         optim_param.append(p)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
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

    writer = SummaryWriter(log_dir='./summary/')
    iter_total = 1
    iter_val_total = 0
    times_val = 0
    for epoch in range(max_epoch):
        for iter, (image, label) in enumerate(train_loader):
            image = image.cuda()
            label = label.cuda()
            logits = model(image)
            loss = loss_function(logits, label)
            print("epoch {}/{}, iter {}/{}, loss = {}".format(epoch + 1, max_epoch,
                                                              iter + 1, train_mini_batch_number, loss))
            writer.add_scalar(tag="loss/train", scalar_value=loss, global_step=iter_total - 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter_total % val_each_iter == 0:
                print("its time to val")
                correct_count = 0
                with torch.no_grad():
                    model.eval()
                    for val_iter, (val_image, val_label) in enumerate(val_loader):
                        val_image = val_image.cuda()
                        val_label = val_label.cuda()
                        val_logits = model(val_image)
                        val_loss = loss_function(val_logits, val_label)
                        correct_count += torch.sum(val_label == torch.argmax(val_logits, dim=1, keepdim=False))
                        print("val, iter {}/{}, loss = {}".format(val_iter + 1, val_mini_batch_number, val_loss))
                        writer.add_scalar(tag="loss/val", scalar_value=val_loss, global_step=iter_val_total)
                        iter_val_total += 1
                    acc = correct_count / (val_mini_batch_number * batch_size)
                    print("val acc= {}".format(acc))
                    writer.add_scalar(tag="val_acc", scalar_value=acc, global_step=times_val)
                    times_val += 1
                    model.train()

            iter_total += 1
    pass
    writer.close()
    torch.save(model, 'emoji_filter_net.pth')


def make_dist_dir(dist_path):
    for i in range(CLASSES_COUNT):
        dist_path_i = os.path.join(dist_path, CLASSES_FOLDER[i])
        if not os.path.exists(dist_path_i):
            os.mkdir(dist_path_i)


def data_cleaning(src_path, dist_path):
    make_dist_dir(dist_path)

    model = torch.load(model_name)
    model.eval()
    inference_transforms = transforms.Compose([
        transforms.Resize([resize_h, resize_w]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    import csv
    with open(os.path.join(dist_path, 'data_cleaning.csv'), 'w+', newline='') as csv_file:
        writer = csv.writer(csv_file)

        for fpathe, dirs, fs in os.walk(src_path):
            for f in fs:
                filename = os.path.join(fpathe, f)
                print(filename)
                file_subfix = os.path.splitext(filename)[-1].replace('.', '')
                if file_subfix in EXEMPT_SUFFIX:
                    print("skip {}".format(filename))
                    continue
                ground_truth = get_label_from_filename(filename)
                try:
                    img = Image.open(filename).convert("RGB")
                    with torch.no_grad():
                        input_tensor = inference_transforms(img).unsqueeze(dim=0).cuda()
                        infer_logits = model(input_tensor)
                        class_id = torch.argmax(infer_logits, dim=1, keepdim=False).cpu().numpy()[0]

                    if ground_truth == class_id:
                        print("{} Classification is correct! ".format(filename))
                    else:
                        writer.writerow([filename, ground_truth, class_id])
                        if class_id not in range(CLASSES_COUNT):
                            raise RuntimeError("class id overflow in file {}!".format(filename))
                        else:
                            mov_dist = os.path.join(dist_path, CLASSES_FOLDER[class_id])
                            print("{} Classification is wrong! move to {}".format(filename, mov_dist))
                            mov(filename, mov_dist)
                except:
                    print("exception on {}".format(filename))


def inference(src_path, dist_path):
    make_dist_dir(dist_path)

    model = torch.load(model_name)
    model.eval()
    inference_transforms = transforms.Compose([
        transforms.Resize([resize_h, resize_w]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    for fpathe, dirs, fs in os.walk(src_path):
        for f in fs:
            filename = os.path.join(fpathe, f)
            print(filename)
            file_subfix = os.path.splitext(filename)[-1].replace('.', '')
            if file_subfix in EXEMPT_SUFFIX:
                print("skip {}".format(filename))
                mov(filename, os.path.join(dist_path, CLASSES_FOLDER[0]))
                continue

            img = Image.open(filename)
            with torch.no_grad():
                input_tensor = inference_transforms(img)
                infer_logits = model(input_tensor)
                class_id = torch.argmax(infer_logits, dim=1, keepdim=False)
                print("class id = {} for {}".format(class_id, filename))

            if class_id not in range(CLASSES_COUNT):
                raise RuntimeError("class id overflow in file {}!".format(filename))
            else:
                mov_dist = os.path.join(dist_path, CLASSES_FOLDER[class_id])
                mov(filename, mov_dist)
    pass


if __name__ == "__main__":
    train_and_val('E:/training_data/dataset.json', 'E:/val_data/dataset.json')
    # data_cleaning(src_path='E:/training_data', dist_path='E:/clean_training_data')