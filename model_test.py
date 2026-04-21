import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from scipy.interpolate import griddata

# 假设所有 .txt 文件在 data_dir 目录下
train_data_dir = r"E:\data_various_width\train_data"
# 在文件开头导入部分后添加设备检测
device = torch.device("cuda" if torch.cuda.is_available() else"cpu")
# 确保这行代码已执行并显示正确设备
print(f"Using device: {device}")  # 应显示"cuda"如果使用GPU

# 修改后的数据加载函数：仅保留单次x坐标，提取y数据矩阵
def load_data(data_dir):
    filenames = os.listdir(data_dir)
    x_data = None  # 存储统一的x坐标
    y_data = []  # 存储所有样本的y坐标
    labels = []

    for filename in filenames:
        if filename.endswith(".txt"):
            # 提取标签（保持原逻辑）
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", filename)
            label = [float(num) for num in numbers[:4]]
            labels.append(label)

            # 读取文件数据
            file_path = os.path.join(data_dir, filename)
            curve_data = np.loadtxt(file_path, delimiter=",")

            # 检测或存储x坐标
            if x_data is None:
                x_data = curve_data[:, 0]
            y = curve_data[:, 1]
            y_data.append(y)

            # 转换为NumPy数组（形状为 [样本数, 数据点数]）
    y_data = np.array(y_data)
    labels = np.array(labels)
    return x_data, y_data, labels  # 返回x坐标、y数据矩阵和标签矩阵


# === 改进1：数据标准化 ===
class InverseDataset(Dataset):
    def __init__(self, labels, y_data, label_stats=None, y_stats=None):
        self.labels = labels
        self.y_data = y_data

        # 标准化标签数据
        if label_stats is None:
            self.label_mean = np.mean(labels, axis=0)
            self.label_std = np.std(labels, axis=0) + 1e-8
        else:
            self.label_mean, self.label_std = label_stats

        # 标准化y数据
        if y_stats is None:
            self.y_mean = np.mean(y_data, axis=0)
            self.y_std = np.std(y_data, axis=0) + 1e-8
        else:
            self.y_mean, self.y_std = y_stats

        self.labels_norm = (labels - self.label_mean) / self.label_std
        self.y_norm = (y_data - self.y_mean) / self.y_std

    def __len__(self):
        return len(self.labels)

    def denormalize_y(self, y_norm):
        return y_norm * self.y_std + self.y_mean

    def __getitem__(self, idx):
        label = torch.tensor(self.labels_norm[idx], dtype=torch.float32).to(device)  # 直接移动到设备
        y = torch.tensor(self.y_norm[idx], dtype=torch.float32).to(device)
        return label, y


# 加载数据（注意返回值变化）
x_coords, y_train, labels_train = load_data(train_data_dir)
# 划分数据集时处理y_data而非完整数据
labels_train, labels_val, y_train, y_val = train_test_split(labels_train, y_train, test_size=0.1, random_state=42)
# 初始化数据集时传递标准化参数
train_dataset = InverseDataset(labels_train, y_train)
val_dataset = InverseDataset(labels_val, y_val,
                           (train_dataset.label_mean, train_dataset.label_std),
                           (train_dataset.y_mean, train_dataset.y_std))


# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


# === 改进2：网络结构增强（新增残差模块）===
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # 更平滑的激活函数

        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out += residual
        out = self.activation(out)
        return out


class EnhancedModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            # ResidualBlock(512, 512),
            # ResidualBlock(512, 512),
            nn.Linear(512, output_size)
            # ResidualBlock(512, 1024),
            # ResidualBlock(1024, 1024),
            # ResidualBlock(1024, 1024),
            # ResidualBlock(1024, 1024),
            # ResidualBlock(1024, 1024),
            # nn.Linear(1024, output_size)
        )

    def forward(self, x):
        return self.net(x)


# 修改模型输出尺寸为y值的长度
input_size = labels_train.shape[1]  # 标签维度不变
output_size = y_train.shape[1]  # 输出等于每个曲线的数据点数

# === 改进3：优化器配置 ===
# 定义模型参数
lr = 0.001
wd = 1e-5
block_type = 'block200'
dropout = 0.2  # 从ResidualBlock类中的默认参数获取
max_epochs = 2000

# 先定义file_prefix变量，但不包含epoch信息，后面会更新
file_prefix = None
model = EnhancedModel(input_size, output_size).to(device)  # 确保模型在GPU上
optimizer = optim.AdamW(model.parameters(),
                       lr=lr,
                       weight_decay=wd)  # L2正则化
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=15, factor=0.5)
# 在模型定义之后、训练循环之前定义损失函数
criterion = nn.MSELoss().to(device)
# === 在模型定义之后添加R²计算函数 ===
def r2_score(y_true, y_pred):
    y_true = y_true.to(device)  # 确保输入张量在正确设备
    y_pred = y_pred.to(device)
    ss_res = torch.sum((y_true - y_pred)**2)
    ss_tot = torch.sum((y_true - torch.mean(y_true))**2)
    return 1 - (ss_res / (ss_tot + 1e-8))

# === 修改训练循环部分 ===
epochs = 2000
max_grad_norm = 1.0
train_losses = []
val_losses = []
train_r2 = []  # 新增：记录训练R²
val_r2 = []    # 新增：记录验证R²

# === 添加早停机制参数 ===
early_stopping_patience = 40  # 如果验证损失在连续50个epoch内没有改善，则停止训练
best_val_loss = float('inf')  # 初始化最佳验证损失为无穷大
counter = 0  # 早停计数器
best_model_state = None  # 用于保存最佳模型状态

for epoch in range(epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    train_r2_batch = 0.0
    for labels, curves in train_loader:
        labels, curves = labels.to(device), curves.to(device)  # 数据移到GPU
        optimizer.zero_grad()
        outputs = model(labels)
        loss = criterion(outputs, curves)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        train_loss += loss.item()
        train_r2_batch += r2_score(curves, outputs).item()
    
    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_r2_batch = 0.0
    with torch.no_grad():
        for labels, curves in val_loader:
            labels, curves = labels.to(device), curves.to(device)  # 数据移到GPU
            outputs = model(labels)
            val_loss += criterion(outputs, curves).item()
            val_r2_batch += r2_score(curves, outputs).item()
    
    # 记录所有指标
    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    train_r2.append(train_r2_batch / len(train_loader))
    val_r2.append(val_r2_batch / len(val_loader))
    print(
        f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
    # 学习率调度和模型保存逻辑保持不变
    scheduler.step(val_loss)
    
    # === 实现早停机制 ===
    current_val_loss = val_losses[-1]
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        counter = 0  # 重置计数器
        # 保存最佳模型状态
        best_model_state = {
            'model_state_dict': model.state_dict(),
            'label_mean': torch.tensor(train_dataset.label_mean),
            'label_std': torch.tensor(train_dataset.label_std),
            'y_mean': train_dataset.y_mean,
            'y_std': train_dataset.y_std,
            'epoch': epoch + 1,
            'best_val_loss': best_val_loss
        }
        print(f"  验证损失改善，当前最佳验证损失: {best_val_loss:.4f}")
    else:
        counter += 1
        print(f"  验证损失未改善，早停计数器: {counter}/{early_stopping_patience}")
        if counter >= early_stopping_patience:
            print(f"早停触发！在第 {epoch + 1} 个epoch停止训练")
            break

# 训练结束后，加载并保存最佳模型
# if best_model_state is not None:
#     torch.save(best_model_state, r'E:\model\rpa_block3_lr0.001_dpt0.2_wd1e-5_f0.5_best.pth')
#     print(f"最佳模型已保存，验证损失: {best_model_state['best_val_loss']:.4f}, 保存路径: E:\model")
# else:
#     # 如果没有触发早停，则保存最后一个模型
#     torch.save({
#                 'model_state_dict': model.state_dict(),
#                 'label_mean': torch.tensor(train_dataset.label_mean),
#                 'label_std': torch.tensor(train_dataset.label_std),
#                 'y_mean': train_dataset.y_mean,
#                 'y_std': train_dataset.y_std
#            }, r'E:\model\rpa_block3_lr0.001_dpt0.2_wd1e-5_f0.5_epoch2000.pth')

# 获取最佳验证损失对应的epoch值
if best_model_state is not None:
    best_epoch = best_model_state['epoch']
else:
    # 如果没有触发早停，则使用实际训练的epoch数
    best_epoch = len(train_losses)
    
# 构建文件名前缀，使用最佳验证损失的epoch值
file_prefix = f'rpa_{block_type}_lr{lr}_dpt{dropout}_wd{wd}_epoch{best_epoch}'

# # 绘制训练和验证损失曲线
# plt.figure(figsize=(12, 6))
# # 使用实际完成的epoch数量，而不是完整的epochs值
# plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
# plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.grid(True)
# plt.show()
#
# plt.figure(figsize=(12, 6))
# plt.plot(range(1, len(train_r2)+1), train_r2, label='Train R²')
# plt.plot(range(1, len(val_r2)+1), val_r2, label='Validation R²')
# plt.xlabel('Epoch')
# plt.ylabel('R² Score')
# plt.title('R² Score During Training')
# plt.legend()
# plt.grid(True)
# plt.show()


# 新增代码：导出训练数据到txt文件
def save_training_data(filename, train_data, val_data):
    with open(filename, 'w') as f:
        f.write('Epoch,Train,Validation\n')
        for epoch in range(len(train_data)):
            f.write(f'{epoch+1},{train_data[epoch]},{val_data[epoch]}\n')

# 保存损失数据
save_training_data(f'{file_prefix}_loss.txt', train_losses, val_losses)
# 保存R²数据
save_training_data(f'{file_prefix}_r2.txt', train_r2, val_r2)

print(f"训练数据已保存到{file_prefix}_loss.txt和{file_prefix}_r2.txt")

# === 新增：对验证集进行热力图分析和正确率计算 ===
# 计算均方根误差
def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# 生成曲线函数
def generate_curve(model, input_labels, x_coords, stats):
    labels_norm = (input_labels - stats['label_mean']) / stats['label_std']
    input_tensor = torch.tensor(labels_norm, dtype=torch.float32).to(device)

    with torch.no_grad():
        y_pred_norm = model(input_tensor).cpu().numpy()

    y_pred = y_pred_norm * stats['y_std'] + stats['y_mean']
    return np.column_stack((x_coords, y_pred.squeeze()))

# 创建统计信息字典
stats = {
    'label_mean': train_dataset.label_mean,
    'label_std': train_dataset.label_std,
    'y_mean': train_dataset.y_mean,
    'y_std': train_dataset.y_std
}

# 设置模型为评估模式
model.eval()

# 用于统计正确率
total_correct = 0
total_samples = len(labels_val)

# 遍历验证集数据
for i in range(total_samples):
    results = []
    original_label = labels_val[i]
    original_y = y_val[i]
    
    # 为当前样本生成热力图数据
    # 第一个label加减10，间隔0.2
    for delta1 in np.arange(-10, 10.1, 0.2):
        # 第二个label加减10，间隔0.2
        for delta2 in np.arange(-10, 10.1, 0.2):
            modified_label = original_label.copy()
            modified_label[0] += delta1
            modified_label[1] += delta2
            
            # 生成曲线
            generated_curve = generate_curve(model, np.array([modified_label]), x_coords, stats)
            generated_y = generated_curve[:, 1]
            
            # 计算RMSE
            rmse = calculate_rmse(original_y, generated_y)
            
            # 保存结果
            results.append({
                'label1': modified_label[0],
                'label2': modified_label[1],
                'label3': modified_label[2],
                'label4': modified_label[3],  # 添加label4
                'rmse': rmse
            })
    
    # 转换为numpy数组方便处理
    label1_values = np.array([r['label1'] for r in results])
    label2_values = np.array([r['label2'] for r in results])
    rmse_values = np.array([r['rmse'] for r in results])

    # 创建网格数据
    xi = np.linspace(min(label1_values), max(label1_values), 200)
    yi = np.linspace(min(label2_values), max(label2_values), 200)
    zi = griddata((label1_values, label2_values), rmse_values, (xi[None, :], yi[:, None]), method='cubic')

    # 绘制等高线并获取等高线对象
    plt.figure(figsize=(10, 8))
    contour = plt.contour(xi, yi, zi, levels=20, colors='black', linewidths=1)

    # 获取所有等高线层级
    contour_levels = contour.levels
    sorted_levels = np.unique(np.sort(contour.levels))
    
    # 初始化预测区间标志
    is_within_range = False
    
    if len(sorted_levels) >= 2:
        target_level = sorted_levels[1]  # 第二小层级
        print(f"第二层等高线值: {target_level:.4f}")

        # 收集所有路径点
        all_points = []
        for level_idx, level in enumerate(contour_levels):
            if np.isclose(level, target_level):
                for path in contour.allsegs[level_idx]:
                    if path.size > 0:
                        all_points.extend(path)

        # 转换为numpy数组并计算范围
        if len(all_points) > 0:
            points = np.array(all_points)
            # 原始标签值
            original_label1 = original_label[0]
            original_label2 = original_label[1]
            original_label3 = original_label[2]
            original_label4 = original_label[3]  # 获取label4的值
            
            # 计算label1相关数据
            label1_min, label1_max = np.min(points[:, 0]), np.max(points[:, 0])
            label1_mean = (label1_min + label1_max) / 2
            label1_range_diff = label1_max - label1_mean
            label1_mean_diff = original_label1 - label1_mean
            
            # 计算label2相关数据
            label2_min, label2_max = np.min(points[:, 1]), np.max(points[:, 1])
            label2_mean = (label2_min + label2_max) / 2
            label2_range_diff = label2_max - label2_mean
            label2_mean_diff = original_label2 - label2_mean
            
            # 判断真实值是否在预测区间内
            if label1_min <= original_label1 <= label1_max and label2_min <= original_label2 <= label2_max:
                is_within_range = True
                total_correct += 1
            
            # 构建输出内容，包含label4
            output_content = f'''
            原始label1: {original_label1:.2f}eV label1范围: [{label1_min:.2f}eV, {label1_max:.2f}eV] label1均值: {label1_mean:.2f}eV 范围差值: {label1_range_diff:.2f}eV 与原始值差值: {label1_mean_diff:.2f}eV
            原始label2: {original_label2:.2f}gcc label2范围: [{label2_min:.2f}gcc, {label2_max:.2f}gcc] label2均值: {label2_mean:.2f}gcc 范围差值: {label2_range_diff:.2f}gcc 与原始值差值: {label2_mean_diff:.2f}gcc 
            label3: {original_label3:.2f}degree
            label4: {original_label4:.2f}eV  # 添加label4的显示
            真实值是否在预测区间内: {'是' if is_within_range else '否'}
{'='*40} '''
            
            # 输出到控制台和文件
            print(output_content)
            with open(f'error_{file_prefix}.txt', 'a') as f:
                f.write(output_content)
    else:
        print(f"第{i}组验证数据等高线层级不足")
        with open(f'error_{file_prefix}.txt', 'a') as f:
            f.write(f"第{i}组验证数据等高线层级不足\n{'='*40}\n")
    
    plt.close()  # 关闭图形但不保存

# 计算并输出正确率
accuracy = total_correct / total_samples * 100
print(f"验证集总正确率: {accuracy:.2f}% ({total_correct}/{total_samples})")
# with open('validation_accuracy.txt', 'w') as f:
#     f.write(f"验证集总正确率: {accuracy:.2f}% ({total_correct}/{total_samples})\n")