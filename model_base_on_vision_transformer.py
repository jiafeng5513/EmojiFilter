import torch
import torch.nn as nn
import torchvision.models as models
from utils import *


class EmojiNet_ResNet(nn.Module):
    def __init__(self, num_class):
        super(EmojiNet_ResNet, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # IMAGENET1K_V1: 76.130, IMAGENET1K_V2: 80.858, DEFAULT
        backbone_output_features = list(self.backbone.children())[-1].out_features
        self.fc1 = nn.Linear(backbone_output_features, 512 - len(Multimodal_features))
        self.bn1d = nn.BatchNorm1d(num_features=512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, num_class)
        # self.fc5 = nn.Linear(64, 32)
        # self.fc6 = nn.Linear(32, 16)
        # self.fc7 = nn.Linear(16, num_class)

    def forward(self, img, multimodal_feature):
        # img [B, C=3, resize_h, resize_w]
        # multimodal_feature [B, len(Multimodal_features)]
        x = self.backbone(img)  # out_features = 1000
        x = self.fc1(x)
        # multimodal_feature = multimodal_feature * 0.001
        x = torch.concat([x, multimodal_feature], dim=1)
        x = self.bn1d(x)
        x = torch.relu(x)
        x = self.fc2(x)
        # x = torch.relu(x)
        x = self.fc3(x)
        # x = torch.relu(x)
        x = self.fc4(x)
        x = self.fc5(x)
        # x = self.fc6(x)
        # x = self.fc7(x)
        output = torch.nn.functional.softmax(x, dim=1)
        return output


def get_model_ResNet():
    model = EmojiNet_ResNet(num_class=CLASSES_COUNT)
    freezd_layer = [model.backbone.conv1, model.backbone.bn1, model.backbone.relu, model.backbone.maxpool,
                    model.backbone.layer1, model.backbone.layer2, model.backbone.layer3]
    no_freeze_layer = [model.backbone.layer4, model.backbone.maxpool, model.backbone.fc,
                       model.fc1, model.fc2, model.fc3, model.fc4, model.fc5, model.bn1d]
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
