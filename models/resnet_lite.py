# models/resnet_lite.py

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """
    基础残差块 (Residual Block)
    包含两次 3x3 卷积，以及一条跳跃连接 (Shortcut)
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # 第一层卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 第二层卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 跳跃连接 (Shortcut)
        self.shortcut = nn.Sequential()
        # 如果维度变了（比如 stride=2 导致宽高中半，或者通道数翻倍），
        # 需要用 1x1 卷积调整 Shortcut 的维度，以便能和卷积的输出相加
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x) # 保存输入
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 核心：将输入直接加到输出上 (Residual Connection)
        out += identity
        out = self.relu(out)
        
        return out

class ResNetLite(nn.Module):
    """
    为 CIFAR-10 (32x32) 定制的简化版 ResNet。
    去掉了 ImageNet 版本中开头的 7x7 卷积和 MaxPool，保留更多的空间信息。
    """
    def __init__(self, num_classes=10):
        super(ResNetLite, self).__init__()
        
        # 初始特征提取： 32x32x3 -> 32x32x16
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        # 残差层 (Layers)
        # Stage 1: 输出 32x32x16
        self.layer1 = self._make_layer(16, stride=1)
        # Stage 2: 输出 16x16x32
        self.layer2 = self._make_layer(32, stride=2)
        # Stage 3: 输出 8x8x64
        self.layer3 = self._make_layer(64, stride=2)
        
        # 全局平均池化 (Global Average Pooling) -> 把 8x8 压成 1x1
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, out_channels, stride):
        layer = BasicBlock(self.in_channels, out_channels, stride)
        self.in_channels = out_channels
        return layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

class ResNetLite_Dropout(nn.Module):
    """
    结合了 ResNet 架构优势与 Dropout 正则化的综合版本。
    """
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(ResNetLite_Dropout, self).__init__()
        
        # 初始特征提取
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        # 残差层
        self.layer1 = self._make_layer(16, stride=1)
        self.layer2 = self._make_layer(32, stride=2)
        self.layer3 = self._make_layer(64, stride=2)
        
        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # --- 核心改动：在分类器前加入 Dropout ---
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, out_channels, stride):
        # 复用你之前文件里写好的 BasicBlock
        layer = BasicBlock(self.in_channels, out_channels, stride)
        self.in_channels = out_channels
        return layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        
        # 经过 Dropout 后再输出给全连接层
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

def get_activation(act_name):
    """动态获取激活函数"""
    if act_name == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_name == 'LeakyReLU':
        return nn.LeakyReLU(inplace=True)
    elif act_name == 'GELU':
        return nn.GELU()
    else:
        raise ValueError("Unsupported activation!")

class BasicBlock_Dynamic(nn.Module):
    """支持动态激活函数的残差块"""
    def __init__(self, in_channels, out_channels, stride=1, act_name='ReLU'):
        super(BasicBlock_Dynamic, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = get_activation(act_name)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = get_activation(act_name)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.act2(out)
        return out
