import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision # 用于后续卷积核可视化
from utils.nn import LabelSmoothingCrossEntropy

from data.loaders import get_cifar_loader
from models.resnet_lite import ResNetLite_experiment
from models.resnet_cifar import resnet20

BATCH_SIZE = 128
EPOCHS = 30
SAVE_DIR = './best_models/res_cifar'  
RESULTS_DIR = './results/res_cifar'

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = get_cifar_loader(batch_size=BATCH_SIZE, train=True)
test_loader = get_cifar_loader(batch_size=BATCH_SIZE, train=False)

# 1. Basket
EXPERIMENTS = {
    # --- 基准线 (Baseline) ---
    "Base_16F_SGD_Cosine": {"filters": 16, "act": "ReLU", "optim": "SGD", "lr": 0.05, "wd": 5e-4, "loss": "CE", "dropout": 0.1, "scheduler": "Cosine"},
    
    # --- 1. 模型容量 (只变 Filters) ---
    "Wider_32F": {"filters": 32, "act": "ReLU", "optim": "SGD", "lr": 0.05, "wd": 5e-4, "loss": "CE", "dropout": 0.1, "scheduler": "Cosine"},
    "Wider_64F": {"filters": 64, "act": "ReLU", "optim": "SGD", "lr": 0.05, "wd": 5e-4, "loss": "CE", "dropout": 0.1, "scheduler": "Cosine"},
    
    # --- 2. 正则化策略 (严格单变量) ---
    # 只改变 Dropout，保持 wd=5e-4
    "Reg_Dropout_0.3": {"filters": 16, "act": "ReLU", "optim": "SGD", "lr": 0.05, "wd": 5e-4, "loss": "CE", "dropout": 0.3, "scheduler": "Cosine"},
    # 只改变 WD，保持 Dropout=0.1
    "Reg_HighWeightDecay": {"filters": 16, "act": "ReLU", "optim": "SGD", "lr": 0.05, "wd": 1e-2, "loss": "CE", "dropout": 0.1, "scheduler": "Cosine"},
    
    # --- 3. 现代组件 (只变 Act 或 Loss) ---
    "Modern_GELU": {"filters": 16, "act": "GELU", "optim": "SGD", "lr": 0.05, "wd": 5e-4, "loss": "CE", "dropout": 0.1, "scheduler": "Cosine"},
    "Modern_LabelSmooth": {"filters": 16, "act": "ReLU", "optim": "SGD", "lr": 0.05, "wd": 5e-4, "loss": "SmoothCE", "dropout": 0.1, "scheduler": "Cosine"},
    
    # --- 4. 优化器与调度器 (严格单变量隔离) ---
    # 只变优化器(及其对应的基础学习率)，保留 Cosine 和 wd=5e-4
    "Opt_Adam_Cosine": {"filters": 16, "act": "ReLU", "optim": "Adam", "lr": 0.001, "wd": 5e-4, "loss": "CE", "dropout": 0.1, "scheduler": "Cosine"},
    # 只变调度器为 StepLR，保留 SGD 和 lr=0.05
    "Sched_SGD_StepLR": {"filters": 16, "act": "ReLU", "optim": "SGD", "lr": 0.05, "wd": 5e-4, "loss": "CE", "dropout": 0.1, "scheduler": "Step"}
}

# 绘图分组 
PLOT_GROUPS = {
    "1_Capacity_Scaling": ["Base_16F_SGD_Cosine", "Wider_32F", "Wider_64F"],
    "2_Regularization": ["Base_16F_SGD_Cosine", "Reg_Dropout_0.3", "Reg_HighWeightDecay"],
    "3_Modern_Components": ["Base_16F_SGD_Cosine", "Modern_GELU", "Modern_LabelSmooth"],
    "4_Scheduler_Impact": ["Opt_Adam_Cosine", "Sched_SGD_StepLR", "Base_16F_SGD_Cosine"]
}

def run_single_experiment(exp_name, config):
    print(f"\n[{exp_name}] Started...")
    model = resnet20().to(device)
    
    # --- 优化器配置 ---
    if config["optim"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
    elif config["optim"] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=config["wd"])
        
    # --- 调度器配置 ---
    if config["scheduler"] == "Cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    elif config["scheduler"] == "Step":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.1)
    else:
        scheduler = None

    # --- Loss 配置 ---
    if config["loss"] == "SmoothCE":
        # 如果你没抽离到 utils，请在这里手写之前的 LabelSmoothing 逻辑
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        # criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # 仅限 PyTorch 1.10+
    else:
        criterion = nn.CrossEntropyLoss()

    train_losses, test_accuracies = [], []
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        # 训练
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
            
        # 评估
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                _, predicted = torch.max(model(inputs).data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        test_accuracies.append(acc)
        
        # 步进调度器
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = config["lr"]
        
        # 保存最佳权重
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'{exp_name}_best.pth'))
            
        print(f"Epoch {epoch+1} | LR: {current_lr:.5f} | Train Loss: {epoch_loss:.4f} | Test Acc: {acc:.2f}%")
            
    return train_losses, test_accuracies

# 运行所有实验
results = {}
for name, cfg in EXPERIMENTS.items():
    results[name] = run_single_experiment(name, cfg)

# 画图逻辑
colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # 蓝，橙，绿
for group_name, exp_list in PLOT_GROUPS.items():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, exp_name in enumerate(exp_list):
        if exp_name in results:
            train_losses, test_accs = results[exp_name]
            ax1.plot(train_losses, label=exp_name, color=colors[idx], linewidth=2)
            ax2.plot(test_accs, label=exp_name, color=colors[idx], linewidth=2)
            
    ax1.set_title(f'{group_name} - Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2.set_title(f'{group_name} - Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{group_name}.png'), dpi=200)
    plt.close()
print("\nAll tasks completed! Plots and best .pth models are saved.")