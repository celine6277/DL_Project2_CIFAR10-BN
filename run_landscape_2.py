import os
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn.utils import parameters_to_vector, vector_to_parameters

# 导入模型和数据
from models.vgg import VGG_A, VGG_A_BatchNorm
from data.loaders import get_cifar_loader

# ==========================================
# 全局配置
# ==========================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FIGURES_PATH = './results/task2'
os.makedirs(FIGURES_PATH, exist_ok=True)

# 探索步长集合
TEST_LRS = [1e-3, 2e-3, 1e-4, 5e-4]
# 用于真实优化的基准学习率
BASE_LR = 0.01 
EPOCHS = 10  # 测绘模式下计算量是正常的数倍，建议先跑 10 轮验证趋势

def get_grad_vector(model):
    """将模型当前的所有梯度展平为一个一维向量"""
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    return parameters_to_vector(grads)

def train_and_survey(model_class, train_loader, model_name):
    print(f"\n========== Surveying Landscape for {model_name} ==========")
    model = model_class().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # 存储指标的容器
    metrics = {
        'loss_max': [], 'loss_min': [],
        'grad_dist_max': [], 'grad_dist_min': [],
        'beta_max': []
    }
    
    model.train()
    step_count = 0
    
    for epoch in range(EPOCHS):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # -------------------------------------------------
            # 1. 基准状态 (Base State) 计算
            # -------------------------------------------------
            optimizer.zero_grad()
            outputs = model(inputs)
            base_loss = criterion(outputs, targets)
            base_loss.backward()
            
            # 提取基准梯度向量 g_t
            g_t = get_grad_vector(model)
            
            # 保存此时的干净权重状态，用于后续恢复
            base_state_dict = copy.deepcopy(model.state_dict())
            
            # -------------------------------------------------
            # 2. 局部地形试探 (Peek Ahead)
            # -------------------------------------------------
            step_losses = []
            step_grad_dists = []
            step_betas = []
            
            for lr_test in TEST_LRS:
                # 手动沿着梯度方向走一步: W_{t, \eta} = W_t - \eta * g_t
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            p.data.sub_(p.grad.data * lr_test)
                
                # 在新的权重位置上，使用【同一批数据】进行前向和反向传播
                model.zero_grad()
                outputs_test = model(inputs)
                loss_test = criterion(outputs_test, targets)
                loss_test.backward()
                
                # 提取试探梯度向量 g_{t, \eta}
                g_t_eta = get_grad_vector(model)
                
                # 计算 L2 距离和 Effective Beta
                grad_diff_l2 = torch.norm(g_t - g_t_eta, p=2).item()
                step_size_l2 = torch.norm(lr_test * g_t, p=2).item()
                beta = grad_diff_l2 / step_size_l2 if step_size_l2 > 1e-8 else 0.0
                
                step_losses.append(loss_test.item())
                step_grad_dists.append(grad_diff_l2)
                step_betas.append(beta)
                
                # 恢复模型到基准权重，准备下一次 lr_test 或真实步进
                model.load_state_dict(base_state_dict)
                # 必须恢复基准梯度，否则 optimizer.step() 会使用最后一次试探的梯度
                model.zero_grad()
                base_loss.backward()

            # -------------------------------------------------
            # 3. 记录边界值并执行真实步进
            # -------------------------------------------------
            metrics['loss_max'].append(max(step_losses))
            metrics['loss_min'].append(min(step_losses))
            metrics['grad_dist_max'].append(max(step_grad_dists))
            metrics['grad_dist_min'].append(min(step_grad_dists))
            metrics['beta_max'].append(max(step_betas))
            
            optimizer.step()
            step_count += 1
            
            if step_count % 50 == 0:
                pbar.set_postfix({
                    'L_max': f"{metrics['loss_max'][-1]:.2f}", 
                    'beta': f"{metrics['beta_max'][-1]:.2f}"
                })
                
    return metrics

def plot_metrics(vgg_metrics, bn_metrics):
    print("\n========== Plotting Results ==========")
    steps = np.arange(len(vgg_metrics['loss_max']))
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    # 1. Loss Landscape
    axes[0].fill_between(steps, vgg_metrics['loss_min'], vgg_metrics['loss_max'], color='green', alpha=0.3, label='Standard VGG')
    axes[0].fill_between(steps, bn_metrics['loss_min'], bn_metrics['loss_max'], color='red', alpha=0.5, label='Standard VGG + BatchNorm')
    axes[0].set_title('Loss Landscape', fontsize=14)
    axes[0].set_ylabel('Loss Value')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # 2. Gradient Predictiveness
    axes[1].fill_between(steps, vgg_metrics['grad_dist_min'], vgg_metrics['grad_dist_max'], color='green', alpha=0.3, label='Standard VGG')
    axes[1].fill_between(steps, bn_metrics['grad_dist_min'], bn_metrics['grad_dist_max'], color='red', alpha=0.5, label='Standard VGG + BatchNorm')
    axes[1].set_title('Gradient Predictiveness (||g_t - g_{t, \eta}||_2)', fontsize=14)
    axes[1].set_ylabel('L2 Distance')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    # 3. Beta-Smoothness
    axes[2].plot(steps, vgg_metrics['beta_max'], color='green', alpha=0.5, label='Standard VGG')
    axes[2].plot(steps, bn_metrics['beta_max'], color='red', alpha=0.8, label='Standard VGG + BatchNorm')
    axes[2].set_title('Optimization Landscape Smoothness (Effective \u03B2)', fontsize=14)
    axes[2].set_ylabel('Effective \u03B2')
    axes[2].set_yscale('log')
    axes[2].set_xlabel('Training Steps')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'unified_landscape_metrics.png'), dpi=300)
    plt.close()

def main():
    train_loader = get_cifar_loader(train=True)
    
    vgg_metrics = train_and_survey(VGG_A, train_loader, "Standard VGG")
    bn_metrics = train_and_survey(VGG_A_BatchNorm, train_loader, "VGG + BatchNorm")
    
    plot_metrics(vgg_metrics, bn_metrics)
    print("All tasks completed successfully.")

if __name__ == '__main__':
    main()