import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# 导入你的模型和数据加载器
from models.vgg import VGG_A, VGG_A_BatchNorm
from data.loaders import get_cifar_loader

# ==========================================
# 0. 全局配置与环境准备
# ==========================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FIGURES_PATH = './results/task2'
os.makedirs(FIGURES_PATH, exist_ok=True)

# 指导书 2.3.1 第 1 步：选择一组学习率代表不同的步长
LEARNING_RATES = [1e-3, 2e-3, 1e-4, 5e-4]
EPOCHS = 35  # 为了快速看到结果，跑 5 个 Epoch 即可获取足够的 steps

def set_strict_seed(seed=2026):
    """
    【工程级细节】：极其关键。
    每次实例化新模型前必须调用，确保所有模型在 Step 0 拥有完全相同的初始权重空间。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ==========================================
# 1 & 2. 核心收集器：遍历 LR 并记录 Step Loss
# ==========================================
def collect_step_losses(model_class, lrs, train_loader, model_name):
    """
    运行一组学习率，收集每个模型、每一个 Step 的 Loss
    返回形状: (len(lrs), total_steps) 的 numpy 数组
    """
    print(f"\n========== Starting Evaluation for {model_name} ==========")
    all_lrs_losses = []
    
    for lr in lrs:
        set_strict_seed(2026) # 确保每次遍历的起点地形完全一致
        
        # 实例化网络、优化器、损失函数
        model = model_class().to(DEVICE)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        step_losses = []
        
        model.train()
        for epoch in range(EPOCHS):
            # 简洁的进度条
            pbar = tqdm(train_loader, desc=f"[{model_name}] LR: {lr} | Epoch: {epoch+1}/{EPOCHS}")
            for inputs, targets in pbar:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                # 记录这一步的 Loss
                current_loss = loss.item()
                step_losses.append(current_loss)
                pbar.set_postfix({'Step Loss': f"{current_loss:.4f}"})
                
        all_lrs_losses.append(step_losses)
        
    return np.array(all_lrs_losses)

# ==========================================
# 主执行流 (指导书 2.3.1 第 3 & 第 4 步)
# ==========================================
def main():
    print(f"Using Device: {DEVICE}")
    train_loader = get_cifar_loader(train=True)
    
    # 获取原始 VGG 损失矩阵 (4, num_steps)
    vgg_losses_matrix = collect_step_losses(VGG_A, LEARNING_RATES, train_loader, "Standard VGG")
    
    # 获取带有 BN 的 VGG 损失矩阵 (4, num_steps)
    bn_losses_matrix = collect_step_losses(VGG_A_BatchNorm, LEARNING_RATES, train_loader, "VGG + BatchNorm")
    
    # 指导书 第 3 步：提取 min_curve 和 max_curve
    # Numpy 的 axis=0 表示跨越不同学习率的维度（列向操作）
    vgg_min_curve = np.min(vgg_losses_matrix, axis=0)
    vgg_max_curve = np.max(vgg_losses_matrix, axis=0)
    
    bn_min_curve = np.min(bn_losses_matrix, axis=0)
    bn_max_curve = np.max(bn_losses_matrix, axis=0)
    
    # 构造 X 轴
    total_steps = vgg_losses_matrix.shape[1]
    steps = np.arange(total_steps)
    
    # 指导书 第 4 步：画图与 fill_between
    print("\n========== Generating Loss Landscape Plot ==========")
    plt.figure(figsize=(12, 7))
    
    # 绘制无 BN 的地形 (绿色)
    plt.fill_between(steps, vgg_min_curve, vgg_max_curve, 
                     color='green', alpha=0.3, label='Standard VGG')
    
    # 绘制带 BN 的地形 (红色)
    plt.fill_between(steps, bn_min_curve, bn_max_curve, 
                     color='red', alpha=0.5, label='Standard VGG + BatchNorm')
    
    # 绘制辅助中位线 (可选，为了让图更好看)
    plt.plot(steps, np.mean(vgg_losses_matrix, axis=0), color='green', alpha=0.8, linewidth=0.5)
    plt.plot(steps, np.mean(bn_losses_matrix, axis=0), color='red', alpha=0.8, linewidth=0.5)

    # 图像配置
    plt.title('Loss Landscape: Impact of Batch Normalization', fontsize=16)
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Loss Value', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 限制 y 轴高度，防止由于个别震荡尖峰把图压得太扁
    plt.ylim(0, 3.0) 
    
    save_path = os.path.join(FIGURES_PATH, '2.3.1_loss_landscape.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Mission Accomplished! The landscape plot is saved to {save_path}")

if __name__ == '__main__':
    main()