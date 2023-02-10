import os
import shutil
import json
import torch.nn as nn
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import dataset
import torch.nn.functional as F

# global def
device = torch.device("cpu")  # device
learning_rate = 1e-2          # init learning rate
max_epoch = 50                # loop training set by max_epoch times
batch_size = 2                # mini batch size
val_each_iter = 10                 # val on each val_iter mini batch
resize_w = 512                # resize w result in preprocess
resize_h = 512                # resize h result in preprocess
model_name = 'emoji_filter_net.pth'  # name of model, for save and load
CLASS_0_FOLDER = "camera"
CLASS_1_FOLDER = "screen_shot"
CLASS_2_FOLDER = "emoji"
CLASS_3_FOLDER = "web"
EXEMPT_SUFFIX = ['avi', 'AVI', 'mp4', 'MP4', 'mov', 'MOV', 'raw', 'RAW', 'ARW', 'arw', 'heic', 'json']


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
                correct_count = 0
                for val_iter, (val_image, val_label) in enumerate(val_loader):
                    val_logits = model(val_image)
                    val_loss = loss_function(val_logits, val_label)
                    correct_count += torch.sum(val_label == torch.argmax(val_logits, dim=1, keepdim=False))
                    print("val, iter {}/{}, loss = {}".format(val_iter + 1, val_mini_batch_number, val_loss))
                acc = correct_count / (val_mini_batch_number * batch_size)
                print("val acc= {}".format(acc))

            iter_total += 1
    pass

    torch.save(model, 'emoji_filter_net.pth')


def mov(srcfile, dstpath):
    """
    move file srcfile to dstpath,
    :param srcfile: abs filename
    :param dstpath: dst path
    :return: None
    """
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)  # 创建路径
        shutil.move(srcfile, os.path.join(dstpath, fname))  # 移动文件


def inference(src_path, dist_path):
    if not os.path.exists(os.path.join(dist_path, CLASS_0_FOLDER)):
        os.mkdir(os.path.join(dist_path, CLASS_0_FOLDER))

    if not os.path.exists(os.path.join(dist_path, CLASS_1_FOLDER)):
        os.mkdir(os.path.join(dist_path, CLASS_1_FOLDER))

    if not os.path.exists(os.path.join(dist_path, CLASS_2_FOLDER)):
        os.mkdir(os.path.join(dist_path, CLASS_2_FOLDER))

    if not os.path.exists(os.path.join(dist_path, CLASS_3_FOLDER)):
        os.mkdir(os.path.join(dist_path, CLASS_3_FOLDER))

    model = torch.load(model_name)

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
                continue

            img = Image.open(filename)
            with torch.no_grad():
                input_tensor = inference_transforms(img)
                infer_logits = model(input_tensor)
                class_id = torch.argmax(infer_logits, dim=1, keepdim=False)
                print("class id = {} for {}".format(class_id, filename))

            if class_id == 0:
                mov(filename, CLASS_0_FOLDER)
            elif class_id == 1:
                mov(filename, CLASS_1_FOLDER)
            elif class_id == 2:
                mov(filename, CLASS_2_FOLDER)
            elif class_id == 3:
                mov(filename, CLASS_3_FOLDER)
            else:
                raise RuntimeError("class id overflow! ")



    pass

if __name__ == "__main__":
    train_and_val('./data/dataset.json', './data/dataset.json')

