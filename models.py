from torchvision import models
import torch.nn as nn
from torch import Tensor
import torch

def model_A(num_classes):
    model_resnet = models.resnet18() 
    num_features = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_features, num_classes)
    print(model_resnet)
    return model_resnet


def conv3x3(in_channels: int, out_channels: int, groups: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_channels, out_channels, groups=groups, kernel_size=3, padding=1, bias=False
    )


class MultiPathwayBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pathways: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()

        self.pathways = pathways
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(in_channels, out_channels, groups=pathways)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, groups=pathways)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ModelB(nn.Module):
    def __init__(self, num_classes: int, pathways: int = 32) -> None:
        super().__init__()

        self.pathways = pathways

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.blocks1 = self.blocks_layer(64, 64, 2)
        self.blocks2 = self.blocks_layer(64, 128, 2)
        self.blocks3 = self.blocks_layer(128, 256, 2)
        self.blocks4 = self.blocks_layer(256, 512, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # * 使用 torchvision ResNet 的初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def blocks_layer(
        self, in_channels: int, out_channels: int, n_blocks: int
    ) -> nn.Sequential:
        downsample = None
        if in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            )

        return nn.Sequential(
            # 第一层使用 in_channels 和 downsample 调整输出 size
            MultiPathwayBlock(in_channels, out_channels, self.pathways, downsample),
            *[
                MultiPathwayBlock(out_channels, out_channels, self.pathways)
                for _ in range(n_blocks - 1)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def model_B(num_classes: int, pretrained: bool = False) -> ModelB:
    # your code here
    return ModelB(num_classes)

