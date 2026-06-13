import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import copy
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# 导入你的模型和数据加载器
from models.resnet_cifar import resnet20
from data.loaders import get_cifar_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = './results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# CIFAR-10 类别名称
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ==========================================
# 1. 卷积核可视化 (Visualization of Filters)
# ==========================================
def plot_filters(model_path):
    print("\n--- 1. Generating Filter Visualization ---")
    model = resnet20().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 提取第一层卷积的权重: 形状为 (16, 3, 3, 3)
    weights = model.conv1.weight.data.clone().cpu()
    
    # 归一化到 [0, 1] 范围以便于显示
    w_min, w_max = weights.min(), weights.max()
    weights = (weights - w_min) / (w_max - w_min)
    
    # 将 16 个卷积核拼接成 4x4 的网格
    grid = torchvision.utils.make_grid(weights, nrow=4, normalize=False, padding=1)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0)) # 转换维度为 (H, W, C)
    plt.title("Visualization of First Layer Filters (ResNet-20)")
    plt.axis('off')
    
    save_path = os.path.join(RESULTS_DIR, 'insight_filters.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved to {save_path}")

# ==========================================
# 2. 网络可解释性：混淆矩阵 (Network Interpretation)
# ==========================================
def plot_confusion_matrix(model_path, test_loader):
    print("\n--- 2. Generating Confusion Matrix ---")
    model = resnet20().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Predicting Test Set"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    cm = confusion_matrix(all_targets, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix (Network Interpretation)', fontsize=14)
    
    save_path = os.path.join(RESULTS_DIR, 'insight_confusion_matrix.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved to {save_path}")

# ==========================================
# 3. 损失地形图 (Loss Landscape - 1D Perturbation)
# ==========================================
def plot_loss_landscape_1d(model_path, train_loader):
    print("\n--- 3. Generating 1D Loss Landscape ---")
    model = resnet20().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    criterion = nn.CrossEntropyLoss()
    
    # 提取最优权重
    optimal_weights = [p.data.clone() for p in model.parameters()]
    
    # 生成与权重同维度的随机高斯方向 (Random Direction)
    direction = [torch.randn_like(p) for p in model.parameters()]
    
    # 滤波器级别归一化 (Filter Normalization)，这是顶会论文的标准做法
    for d, w in zip(direction, optimal_weights):
        d.mul_(w.norm() / (d.norm() + 1e-10))
        
    alphas = np.linspace(-1.0, 1.0, 21) # 在最优解前后各走 10 步
    losses = []
    
    # 为了加快速度，我们只用训练集的前 5 个 Batch 来估算 Loss 表面
    mini_batches = []
    for i, (inputs, targets) in enumerate(train_loader):
        if i >= 5: break
        mini_batches.append((inputs.to(device), targets.to(device)))
        
    for alpha in tqdm(alphas, desc="Scanning Landscape"):
        # 将权重扰动: W_new = W_opt + alpha * direction
        for p, w, d in zip(model.parameters(), optimal_weights, direction):
            p.data.copy_(w + alpha * d)
            
        # 计算扰动后的 Loss
        model.eval()
        current_loss = 0.0
        with torch.no_grad():
            for inputs, targets in mini_batches:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                current_loss += loss.item()
                
        losses.append(current_loss / len(mini_batches))
        
    plt.figure(figsize=(8, 6))
    plt.plot(alphas, losses, marker='o', linestyle='-', color='purple', linewidth=2)
    plt.title('1D Loss Landscape around Minimum', fontsize=14)
    plt.xlabel('Perturbation (alpha)', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 标出最优解点
    min_idx = np.argmin(losses)
    plt.axvline(x=alphas[min_idx], color='red', linestyle='--', label=f'Min Loss = {losses[min_idx]:.4f}')
    plt.legend()
    
    save_path = os.path.join(RESULTS_DIR, 'insight_loss_landscape.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved to {save_path}")

# ==========================================
# 执行入口
# ==========================================
if __name__ == '__main__':
    # 假设你刚刚用 ResNet-20 跑出了最好的模型，填入它的路径
    # 注意：请确保这个 .pth 文件存在！
    BEST_MODEL_PATH = './best_models/res_cifar/Base_16F_SGD_Cosine_best.pth' 
    
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"ERROR: Model {BEST_MODEL_PATH} not found. Please update the path.")
    else:
        # 加载数据 (测试集不需要 shuffle，也不要 Data Augmentation)
        test_loader = get_cifar_loader(batch_size=128, train=False)
        train_loader = get_cifar_loader(batch_size=128, train=True) # 用于计算 Loss 表面
        
        # 依次运行三个可视化任务
        plot_filters(BEST_MODEL_PATH)
        plot_confusion_matrix(BEST_MODEL_PATH, test_loader)
        plot_loss_landscape_1d(BEST_MODEL_PATH, train_loader)
        
        print("\nAll insights generated successfully! Check your ./results/ folder.")