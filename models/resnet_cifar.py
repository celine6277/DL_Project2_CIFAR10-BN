# models/resnet_cifar.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CifarBasicBlock(nn.Module):
    """
    CIFAR-10 专用的基础残差块
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(CifarBasicBlock, self).__init__()
        # 第一层卷积
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # 第二层卷积
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 捷径连接 (Shortcut)
        self.shortcut = nn.Sequential()
        # 当下采样发生（stride=2）或通道数翻倍时，使用 1x1 卷积调整输入 X 的尺寸与通道数
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        # 核心数学公式：Y = F(X) + X
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差相加
        out = F.relu(out)
        return out


class ResNetCifar(nn.Module):
    """
    针对 CIFAR-10 (32x32) 定制的标准 ResNet
    默认配置为 [3, 3, 3] 即为 ResNet-20
    """
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetCifar, self).__init__()
        self.in_planes = 16

        # 1. 初始特征提取层 (与 ImageNet 版的 7x7 卷积不同，CIFAR 专用版使用 3x3 卷积，且不加 MaxPool)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # 2. 三个阶段的残差堆叠 (Stage 1 -> Stage 2 -> Stage 3)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        # 3. 分类器部分 (全局平均池化 + 最终线性映射)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)  # 输出形状: [batch, 16, 32, 32]
        out = self.layer2(out)  # 输出形状: [batch, 32, 16, 16]
        out = self.layer3(out)  # 输出形状: [batch, 64, 8, 8]
        
        # 全局平均池化 Global Average Pooling (将 8x8 空间维度压减成 1x1)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    """返回标准 ResNet-20 模型"""
    return ResNetCifar(CifarBasicBlock, [3, 3, 3])

def resnet32():
    """返回标准 ResNet-32 模型"""
    return ResNetCifar(CifarBasicBlock, [5, 5, 5])

def resnet56():
    """返回标准 ResNet-56 模型"""
    return ResNetCifar(CifarBasicBlock, [9, 9, 9])