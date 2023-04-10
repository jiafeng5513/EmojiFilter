import torch
import torch.nn as nn
import torchvision.models as models
from utils import *


class EmojiNet_EfficientNet(nn.Module):
    def __init__(self, num_class):
        super(EmojiNet_EfficientNet, self).__init__()
        # self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # IMAGENET1K_V1: 76.130, IMAGENET1K_V2: 80.858, DEFAULT
        from efficientnet_pytorch import EfficientNet
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        feature = self.backbone._fc.in_features
        self.backbone._fc = nn.Linear(in_features=feature, out_features=num_class, bias=True)
        print(self.backbone)
        # from efficientnet_pytorch import EfficientNet
        # from torch import nn
        # model = EfficientNet.from_pretrained('efficientnet-b5')
        # feature = model._fc.in_features
        # model._fc = nn.Linear(in_features=feature, out_features=50, bias=True)

        # backbone_output_features = list(self.backbone.children())[-1].out_features
        # self.fc1 = nn.Linear(1024, 512 - len(Multimodal_features))
        # self.bn1d = nn.BatchNorm1d(num_features=512)
        # self.fc2 = nn.Linear(512, 512)
        # self.fc3 = nn.Linear(512, 256)
        # self.fc4 = nn.Linear(256, 128)
        # self.fc5 = nn.Linear(128, num_class)
        # self.fc5 = nn.Linear(64, 32)
        # self.fc6 = nn.Linear(32, 16)
        # self.fc7 = nn.Linear(16, num_class)

    def forward(self, img, multimodal_feature):
        # img [B, C=3, resize_h, resize_w]
        # multimodal_feature [B, len(Multimodal_features)]
        x = self.backbone(img)  # out_features = 1000
        # x = self.fc1(x)
        # multimodal_feature = multimodal_feature * 0.001
        # x = torch.concat([x, multimodal_feature], dim=1)
        # x = self.bn1d(x)
        # x = torch.relu(x)
        # x = self.fc2(x)
        # x = torch.relu(x)
        # x = self.fc3(x)
        # x = torch.relu(x)
        # x = self.fc4(x)
        # x = self.fc5(x)
        # x = self.fc6(x)
        # x = self.fc7(x)
        # output = torch.nn.functional.softmax(x, dim=1)
        return x


def get_model_EfficientNet():
    model = EmojiNet_EfficientNet(num_class=CLASSES_COUNT)
    print(model)
    freezd_layer = [model.backbone._conv_stem, model.backbone._bn0]
    for i in range(14):
        freezd_layer.append(model.backbone._blocks._modules[str(i)])

    no_freeze_layer = [model.backbone._blocks._modules['15'], model.backbone._conv_head, model.backbone._bn1,
                       model.backbone._avg_pooling, model.backbone._dropout, model.backbone._fc, model.backbone._swish]
    optim_param = []
    #
    for layer in freezd_layer:
        for p in layer.parameters():
            p.requires_grad = False
    for layer in no_freeze_layer:
        for p in layer.parameters():
            p.requires_grad = True
            optim_param.append(p)

    optimizer = torch.optim.SGD(optim_param, lr=learning_rate)
    step_schedule = torch.optim.lr_scheduler.StepLR(step_size=100, gamma=0.9, optimizer=optimizer)
    return model, optimizer, step_schedule
