# train_task1.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# 导入你自己的模块
from data.loaders import get_cifar_loader
from models.simple_cnn import SimpleCNNBaseline

# 1. 超参数与配置
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 0.001
SAVE_DIR = './best_models'
RESULTS_DIR = './results'

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. 准备数据
print("Loading data...")
train_loader = get_cifar_loader(batch_size=BATCH_SIZE, train=True)
test_loader = get_cifar_loader(batch_size=BATCH_SIZE, train=False)

# 3. 实例化模型、优化器和损失函数
print("Building model...")
model = SimpleCNNBaseline(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 4. 训练与评估逻辑
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    return running_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return 100 * correct / total

# 5. 主循环
train_losses = []
test_accuracies = []
best_acc = 0.0

print("Starting training loop...")
for epoch in range(EPOCHS):
    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
    
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    test_acc = evaluate(model, test_loader, device)
    
    train_losses.append(train_loss)
    test_accuracies.append(test_acc)
    
    print(f"Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    # 保存最佳模型
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'simple_cnn_best.pth'))
        print("--> Saved new best model!")

# 6. 可视化并保存结果
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_losses, label='Train Loss')
ax1.set_title('Training Loss')
ax2.plot(test_accuracies, label='Test Accuracy')
ax2.set_title('Test Accuracy')
plt.savefig(os.path.join(RESULTS_DIR, 'task1_baseline_curve.png'))
print("\nTraining complete! Results saved.")