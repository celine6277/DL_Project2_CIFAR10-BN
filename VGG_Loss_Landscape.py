# 2.2 Loss Landscape 对比：VGG 标准版 vs 带 BatchNorm 的版本
import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.vgg import VGG_A, VGG_A_BatchNorm
from data.loaders import get_cifar_loader

# ==========================================
# 0. 全局配置
# ==========================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FIGURES_PATH = './results/task2'
os.makedirs(FIGURES_PATH, exist_ok=True)

EPOCHS = 35    # 跑 15 轮足以看出 BN 的巨大优势
LEARNING_RATE = 0.01

### Sanity Check!
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)
for X,y in train_loader:
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    img = np.transpose(X[0].numpy(), (1, 2, 0))  
    img = img * 0.5 + 0.5 
    plt.figure(figsize=(2, 2))
    plt.imshow(img)
    plt.title(f'Label: {y[0].item()}')
    plt.savefig(os.path.join(FIGURES_PATH, 'sanity_check.png')) # 保存出来看看
    plt.close() # 极其关键：清空画布
    break

def set_strict_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_accuracy(model, dataloader):
    """计算准确率的小组件"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# ==========================================
# 1. 核心训练流：专注收集 Epoch 级别的数据
# ==========================================
def train_model_for_comparison(model_class, train_loader, val_loader, model_name):
    print(f"\n========== Training {model_name} ==========")
    set_strict_seed(2026) # 保证两个模型初始化的随机性一模一样
    
    model = model_class().to(DEVICE)
    # 使用带 Momentum 的 SGD，这是图像分类的标配
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    epoch_losses = []
    val_accuracies = []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # 记录本轮平均 Loss
        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        
        # 记录本轮验证集准确率
        val_acc = get_accuracy(model, val_loader) * 100 # 转为百分比
        val_accuracies.append(val_acc)
        
        print(f"[{model_name}] Epoch {epoch+1} -> Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
    return epoch_losses, val_accuracies

# ==========================================
# 2. 主函数：运行与同框作图
# ==========================================
def main():
    print(f"Using Device: {DEVICE}")
    train_loader = get_cifar_loader(train=True)
    val_loader = get_cifar_loader(train=False)
    
    # 获取无 BN 的数据
    vgg_losses, vgg_accs = train_model_for_comparison(VGG_A, train_loader, val_loader, "Standard VGG")
    
    # 获取带 BN 的数据
    bn_losses, bn_accs = train_model_for_comparison(VGG_A_BatchNorm, train_loader, val_loader, "VGG + BatchNorm")
    
    # --- 开始画对比图 ---
    print("\n========== Generating Comparison Plot ==========")
    epochs_range = np.arange(1, EPOCHS + 1)
    
    # 设置一个正常的比例 (12, 5) 而不是原代码拉爆的 (15, 3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：Loss 对比
    ax1.plot(epochs_range, vgg_losses, label='Standard VGG', color='#2ca02c', linewidth=2, marker='o')
    ax1.plot(epochs_range, bn_losses, label='VGG + BatchNorm', color='#d62728', linewidth=2, marker='s')
    ax1.set_title('Training Loss Comparison', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Average Loss', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10)
    
    # 右图：Accuracy 对比
    ax2.plot(epochs_range, vgg_accs, label='Standard VGG', color='#2ca02c', linewidth=2, marker='o')
    ax2.plot(epochs_range, bn_accs, label='VGG + BatchNorm', color='#d62728', linewidth=2, marker='s')
    ax2.set_title('Validation Accuracy Comparison', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(FIGURES_PATH, '2.2_performance_comparison.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Mission Accomplished! Plot saved to {save_path}")

if __name__ == '__main__':
    main()