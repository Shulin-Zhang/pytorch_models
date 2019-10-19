# zhangshulin
# zhangslwork@yeah.net
# 2019-10-10


import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity_block(nn.Module):
    def __init__(self, in_channel, channels, groups=32,
                 activation=nn.ReLU(True)):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channel, channels[0], 1)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3,
                               groups=groups, padding=1)
        self.bn2 = nn.BatchNorm2d(channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 1)
        self.bn3 = nn.BatchNorm2d(channels[2])
        self.activation = activation

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = x + shortcut

        x = self.activation(x)

        return x


class Conv_block(nn.Module):
    def __init__(self, in_channel, channels,
                 groups=32, activation=nn.ReLU(True)):
        super().__init__()

        self.max_pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(in_channel, channels[0], 1)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3,
                               groups=groups, padding=1)
        self.bn2 = nn.BatchNorm2d(channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 1)
        self.bn3 = nn.BatchNorm2d(channels[2])

        self.conv_shortcut = nn.Conv2d(in_channel, channels[2], 1)
        self.bn_shortcut = nn.BatchNorm2d(channels[2])

        self.activation = activation

    def forward(self, x):
        x = self.max_pool(x)

        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)

        x = self.conv3(x)
        x = self.bn3(x)

        shortcut = self.conv_shortcut(shortcut)
        shortcut = self.bn_shortcut(shortcut)
        x = x + shortcut

        x = self.activation(x)

        return x


class ResNeXt(nn.Module):
    def __init__(self, class_num, in_channel=3,
                 groups=32, activation=nn.ReLU(True)):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            activation,

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            activation,

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            activation,
        )

        self.layer2 = nn.Sequential(
            Conv_block(64, [64, 64, 256], groups, activation),
            Identity_block(256, [64, 64, 256], groups, activation),
            Identity_block(256, [64, 64, 256], groups, activation),
        )

        self.layer3 = nn.Sequential(
            Conv_block(256, [128, 128, 512], groups, activation),
            Identity_block(512, [128, 128, 512], groups, activation),
            Identity_block(512, [128, 128, 512], groups, activation),
            Identity_block(512, [128, 128, 512], groups, activation),
        )

        self.layer4 = nn.Sequential(
            Conv_block(512, [256, 256, 1024], groups, activation),
            Identity_block(1024, [256, 256, 1024], groups, activation),
            Identity_block(1024, [256, 256, 1024], groups, activation),
            Identity_block(1024, [256, 256, 1024], groups, activation),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, class_num)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x
