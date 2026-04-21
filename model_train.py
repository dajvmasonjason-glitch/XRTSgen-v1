import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scipy.interpolate import griddata
import os

# 数据目录设置
train_data_dir = r"E:\data_various_width\train_data"
test_data_dir = r"E:\data_various_width\test_data"
test_1_data_dir = r"E:\data_various_width\test_data_1"

# 设备检测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 修改后的数据加载函数：仅保留单次x坐标，提取y数据矩阵
def load_data(data_dir):
    filenames = os.listdir(data_dir)
    x_data = None  # 存储统一的x坐标
    y_data = []  # 存储所有样本的y坐标
    labels = []

    for filename in filenames:
        if filename.endswith(".txt"):
            # 提取标签
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

    y_data = np.array(y_data)
    labels = np.array(labels)
    return x_data, y_data, labels  # 返回x坐标、y数据矩阵和标签矩阵


# 数据标准化类
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
        label = torch.tensor(self.labels_norm[idx], dtype=torch.float32).to(device)
        y = torch.tensor(self.y_norm[idx], dtype=torch.float32).to(device)
        return label, y


# 加载训练数据和测试数据
print("加载训练数据...")
x_coords, y_train, labels_train = load_data(train_data_dir)
print("加载测试集数据...")
x_coords_test, y_test, labels_test = load_data(test_data_dir)
print("加载测试集1数据...")
x_coords_test_1, y_test_1, labels_test_1 = load_data(test_1_data_dir)

# 初始化训练集数据集（不再划分验证集）
train_dataset = InverseDataset(labels_train, y_train)

# 初始化测试集数据集
test_dataset = InverseDataset(labels_test, y_test,
                            (train_dataset.label_mean, train_dataset.label_std),
                            (train_dataset.y_mean, train_dataset.y_std))
test_1_dataset = InverseDataset(labels_test_1, y_test_1,
                               (train_dataset.label_mean, train_dataset.label_std),
                               (train_dataset.y_mean, train_dataset.y_std))

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)
test_1_loader = DataLoader(test_1_dataset, batch_size=32)

# 定义模型参数
lr = 0.001
wd = 1e-5
block_type = 'block300'
dropout = 0.2

# 残差模块类
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=dropout):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

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


# 增强模型类
class EnhancedModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            # ResidualBlock(512, 1024),
            # ResidualBlock(1024, 1024),
            # ResidualBlock(1024, 1024),
            # ResidualBlock(1024, 1024),
            nn.Linear(512, output_size)
        )

    def forward(self, x):
        return self.net(x)


# 修改模型输出尺寸
input_size = labels_train.shape[1]
output_size = y_train.shape[1]

# 初始化模型
model = EnhancedModel(input_size, output_size).to(device)
optimizer = optim.AdamW(model.parameters(),
                       lr=lr,
                       weight_decay=wd)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=15, factor=0.5)
criterion = nn.MSELoss().to(device)

# R²计算函数
def r2_score(y_true, y_pred):
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)
    ss_res = torch.sum((y_true - y_pred)**2)
    ss_tot = torch.sum((y_true - torch.mean(y_true))**2)
    return 1 - (ss_res / (ss_tot + 1e-8))

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

# 训练参数设置
epochs = 2000
max_grad_norm = 1.0
train_losses = []
train_r2 = []
test_losses = []  # 记录test集损失（用于早停）
test_r2 = []      # 记录test集R²


# 早停机制参数（基于test集）
early_stopping_patience = 40
best_test_loss = float('inf')
counter = 0
best_model_state = None

# 训练循环
for epoch in range(epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    train_r2_batch = 0.0
    for labels, curves in train_loader:
        optimizer.zero_grad()
        outputs = model(labels)
        loss = criterion(outputs, curves)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        train_loss += loss.item()
        train_r2_batch += r2_score(curves, outputs).item()
    
    # 测试阶段
    model.eval()
    test_loss = 0.0  # test集损失
    test_r2_batch = 0.0  # test集R²
    
    with torch.no_grad():
        # test集评估（用于早停）
        for labels, curves in test_loader:
            outputs = model(labels)
            test_loss += criterion(outputs, curves).item()
            test_r2_batch += r2_score(curves, outputs).item()
    
    # 记录所有指标
    train_losses.append(train_loss / len(train_loader))
    train_r2.append(train_r2_batch / len(train_loader))
    test_losses.append(test_loss / len(test_loader))
    test_r2.append(test_r2_batch / len(test_loader))
    
    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')
    
    # 学习率调度（基于测试损失）
    scheduler.step(test_loss)
    
    # 基于test集的早停机制
    current_test_loss = test_losses[-1]
    if current_test_loss < best_test_loss:
        best_test_loss = current_test_loss
        counter = 0  # 重置计数器
        # 保存最佳模型状态
        best_model_state = {
            'model_state_dict': model.state_dict(),
            'label_mean': torch.tensor(train_dataset.label_mean),
            'label_std': torch.tensor(train_dataset.label_std),
            'y_mean': train_dataset.y_mean,
            'y_std': train_dataset.y_std,
            'epoch': epoch + 1,
            'best_test_loss': best_test_loss
        }
        print(f"  测试损失改善，当前最佳测试损失: {best_test_loss:.4f}")
    else:
        counter += 1
        print(f"  测试损失未改善，早停计数器: {counter}/{early_stopping_patience}")
        if counter >= early_stopping_patience:
            print(f"早停触发！在第 {epoch + 1} 个epoch停止训练")
            break
# 获取最佳测试损失对应的epoch值
if best_model_state is not None:
    best_epoch = best_model_state['epoch']
else:
    best_epoch = len(train_losses)

# 构建文件名前缀
file_prefix = f'rpa_train_{block_type}_lr{lr}_dpt{dropout}_wd{wd}_epoch{best_epoch}'

# 训练结束后，加载并保存最佳模型
if best_model_state is not None:
    torch.save(best_model_state, f'model/best_model_{file_prefix}.pth')
    print(f"最佳模型已保存，测试损失: {best_model_state['best_test_loss']:.4f}")
    # 加载最佳模型权重
    model.load_state_dict(best_model_state['model_state_dict'])
else:
    print("警告：未找到最佳模型状态")


# # 绘制训练和测试损失曲线
# plt.figure(figsize=(12, 6))
# plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
# plt.plot(range(1, len(test_losses)+1), test_losses, label='Test Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Test Loss')
# plt.legend()
# plt.grid(True)
# plt.show()
#
# # 绘制R²曲线
# plt.figure(figsize=(12, 6))
# plt.plot(range(1, len(train_r2)+1), train_r2, label='Train R²')
# plt.plot(range(1, len(test_r2)+1), test_r2, label='Test R²')
# plt.xlabel('Epoch')
# plt.ylabel('R² Score')
# plt.title('R² Score During Training')
# plt.legend()
# plt.grid(True)
# plt.show()

# 保存训练数据到txt文件
def save_training_data(filename, train_data, test_data):
    with open(filename, 'w') as f:
        f.write('Epoch,Train,Test\n')
        for epoch in range(len(train_data)):
            f.write(f'{epoch+1},{train_data[epoch]},{test_data[epoch]}\n')

# 保存损失数据
save_training_data(f'{file_prefix}_loss.txt', train_losses, test_losses)
# 保存R²数据
save_training_data(f'{file_prefix}_r2.txt', train_r2, test_r2)

print(f"训练数据已保存到{file_prefix}_loss.txt和{file_prefix}_r2.txt")

# 创建统计信息字典
stats = {
    'label_mean': train_dataset.label_mean,
    'label_std': train_dataset.label_std,
    'y_mean': train_dataset.y_mean,
    'y_std': train_dataset.y_std
}

# 设置模型为评估模式
model.eval()

# 定义函数来评估测试集并保存等高线数据
def evaluate_test_set(labels_data, y_data, x_coords, set_name, file_prefix):
    print(f"\n评估{set_name}测试集...")
    total_correct = 0
    total_samples = len(labels_data)
    max_ratio = 0.0  # 用于记录最大比值
    ratio_distribution = {  # 用于统计比值分布
        'lt_1pct': 0,    # 小于1%
        'lt_2_5pct': 0,  # 小于2.5%
        'lt_5pct': 0,    # 小于5%
        'lt_10pct': 0,   # 小于10%
        'lt_15pct': 0,   # 小于15%
        'lt_20pct': 0,   # 小于20%
        'gt_20pct': 0    # 大于等于20%
    }
    
    # 打开错误文件用于写入
    error_file = open(f'error_{file_prefix}_{set_name}.txt', 'a')
    
    for i in range(total_samples):
        print(f"处理{set_name}测试集样本 {i+1}/{total_samples}")
        results = []
        original_label = labels_data[i]
        original_y = y_data[i]
        
        # 为当前样本生成热力图数据
        for delta1 in np.arange(-10, 10.1, 0.2):
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
                    'label4': modified_label[3],
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
        
        # 找到最小RMSE对应的label1和label2
        min_rmse_idx = np.argmin(rmse_values)
        min_rmse_label1 = label1_values[min_rmse_idx]
        min_rmse_label2 = label2_values[min_rmse_idx]
        min_rmse = rmse_values[min_rmse_idx]
        
        # 计算差值和比值
        label1_diff = np.abs(min_rmse_label1 - original_label[0])
        label2_diff = np.abs(min_rmse_label2 - original_label[1])
        label1_ratio = label1_diff / (np.abs(original_label[0]) + 1e-8)
        label2_ratio = label2_diff / (np.abs(original_label[1]) + 1e-8)
        
        # 取label1和label2比值中的较大值
        current_max_ratio = max(label1_ratio, label2_ratio)
        
        # 更新比值分布统计
        current_ratio_pct = current_max_ratio * 100  # 转换为百分比
        if current_ratio_pct < 1.0:
            ratio_distribution['lt_1pct'] += 1
        if current_ratio_pct < 2.5:
            ratio_distribution['lt_2_5pct'] += 1
        if current_ratio_pct < 5.0:
            ratio_distribution['lt_5pct'] += 1
        if current_ratio_pct < 10.0:
            ratio_distribution['lt_10pct'] += 1
        if current_ratio_pct < 15.0:
            ratio_distribution['lt_15pct'] += 1
        if current_ratio_pct < 20.0:
            ratio_distribution['lt_20pct'] += 1
        else:
            ratio_distribution['gt_20pct'] += 1
        
        # 更新最大比值
        if current_max_ratio > max_ratio:
            max_ratio = current_max_ratio
        
        # 初始化预测区间标志
        is_within_range = False
        
        if len(sorted_levels) >= 2:
            target_level = sorted_levels[1]  # 第二小层级
            print(f"{set_name}测试集样本 {i+1} 第二层等高线值: {target_level:.4f}")

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
                original_label4 = original_label[3]
                
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
                
                # 构建输出内容，包含最小RMSE和比值信息
                output_content = f'''
                原始label1: {original_label1:.2f}eV label1范围: [{label1_min:.2f}eV, {label1_max:.2f}eV] label1均值: {label1_mean:.2f}eV 范围差值: {label1_range_diff:.2f}eV 与原始值差值: {label1_mean_diff:.2f}eV
                原始label2: {original_label2:.2f}gcc label2范围: [{label2_min:.2f}gcc, {label2_max:.2f}gcc] label2均值: {label2_mean:.2f}gcc 范围差值: {label2_range_diff:.2f}gcc 与原始值差值: {label2_mean_diff:.2f}gcc 
                label3: {original_label3:.2f}degree
                label4: {original_label4:.2f}eV
                真实值是否在预测区间内: {'是' if is_within_range else '否'}
                
                最小RMSE信息:
                最小RMSE值: {min_rmse:.6f}
                最小RMSE对应label1: {min_rmse_label1:.2f}eV, 与原始值差值: {label1_diff:.2f}eV, 比值: {label1_ratio:.6f} ({label1_ratio*100:.2f}%)
                最小RMSE对应label2: {min_rmse_label2:.2f}gcc, 与原始值差值: {label2_diff:.2f}gcc, 比值: {label2_ratio:.6f} ({label2_ratio*100:.2f}%)
                最大比值: {current_max_ratio:.6f} ({current_ratio_pct:.2f}%)
{'='*60} '''
                
                # 输出到控制台和文件
                print(output_content)
                error_file.write(output_content)
        else:
            error_msg = f"{set_name}测试集第{i+1}组数据等高线层级不足\n{'='*60}\n"
            print(error_msg)
            error_file.write(error_msg)
        
        plt.close()  # 关闭图形但不保存
    
    # 关闭错误文件
    error_file.close()
    
    # 计算并输出正确率
    accuracy = total_correct / total_samples * 100
    print(f"\n{set_name}测试集总正确率: {accuracy:.2f}% ({total_correct}/{total_samples})")
    print(f"{set_name}测试集最大比值: {max_ratio:.6f}")
    
    # 保存正确率和最大比值信息，以及比值分布
    with open(f'{set_name}_results_{file_prefix}.txt', 'w') as f:
        f.write(f"{set_name}测试集总正确率: {accuracy:.2f}% ({total_correct}/{total_samples})\n")
        f.write(f"{set_name}测试集最大比值: {max_ratio:.6f} ({max_ratio*100:.2f}%)\n\n")
        f.write(f"比值分布统计:\n")
        f.write(f"小于1%: {ratio_distribution['lt_1pct']} 个 ({ratio_distribution['lt_1pct']/total_samples*100:.2f}%)\n")
        f.write(f"小于2.5%: {ratio_distribution['lt_2_5pct']} 个 ({ratio_distribution['lt_2_5pct']/total_samples*100:.2f}%)\n")
        f.write(f"小于5%: {ratio_distribution['lt_5pct']} 个 ({ratio_distribution['lt_5pct']/total_samples*100:.2f}%)\n")
        f.write(f"小于10%: {ratio_distribution['lt_10pct']} 个 ({ratio_distribution['lt_10pct']/total_samples*100:.2f}%)\n")
        f.write(f"小于15%: {ratio_distribution['lt_15pct']} 个 ({ratio_distribution['lt_15pct']/total_samples*100:.2f}%)\n")
        f.write(f"小于20%: {ratio_distribution['lt_20pct']} 个 ({ratio_distribution['lt_20pct']/total_samples*100:.2f}%)\n")
        f.write(f"大于等于20%: {ratio_distribution['gt_20pct']} 个 ({ratio_distribution['gt_20pct']/total_samples*100:.2f}%)\n")
    
    return accuracy, max_ratio, ratio_distribution

# 分别评估两个测试集
print(f"\n{'='*50}\n开始评估测试集...\n{'='*50}")
test_accuracy, test_max_ratio, test_ratio_dist = evaluate_test_set(labels_test, y_test, x_coords_test, "test", file_prefix)

print(f"\n{'='*50}\n开始评估测试集1...\n{'='*50}")
test_1_accuracy, test_1_max_ratio, test_1_ratio_dist = evaluate_test_set(labels_test_1, y_test_1, x_coords_test_1, "test_1", file_prefix)

# 输出最终结果摘要
print(f"\n{'='*80}")
print("最终评估结果摘要：")
print(f"Test测试集: 正确率 = {test_accuracy:.2f}%, 最大比值 = {test_max_ratio:.6f} ({test_max_ratio*100:.2f}%)")
print(f"Test_1测试集: 正确率 = {test_1_accuracy:.2f}%, 最大比值 = {test_1_max_ratio:.6f} ({test_1_max_ratio*100:.2f}%)")
print(f"\n{'='*80}")
print("Test测试集比值分布统计：")
test_total = len(labels_test)
print(f"小于1%: {test_ratio_dist['lt_1pct']} 个 ({test_ratio_dist['lt_1pct']/test_total*100:.2f}%)")
print(f"小于2.5%: {test_ratio_dist['lt_2_5pct']} 个 ({test_ratio_dist['lt_2_5pct']/test_total*100:.2f}%)")
print(f"小于5%: {test_ratio_dist['lt_5pct']} 个 ({test_ratio_dist['lt_5pct']/test_total*100:.2f}%)")
print(f"小于10%: {test_ratio_dist['lt_10pct']} 个 ({test_ratio_dist['lt_10pct']/test_total*100:.2f}%)")
print(f"小于15%: {test_ratio_dist['lt_15pct']} 个 ({test_ratio_dist['lt_15pct']/test_total*100:.2f}%)")
print(f"小于20%: {test_ratio_dist['lt_20pct']} 个 ({test_ratio_dist['lt_20pct']/test_total*100:.2f}%)")
print(f"大于等于20%: {test_ratio_dist['gt_20pct']} 个 ({test_ratio_dist['gt_20pct']/test_total*100:.2f}%)")
print(f"\n{'='*80}")
print("Test_1测试集比值分布统计：")
test_1_total = len(labels_test_1)
print(f"小于1%: {test_1_ratio_dist['lt_1pct']} 个 ({test_1_ratio_dist['lt_1pct']/test_1_total*100:.2f}%)")
print(f"小于2.5%: {test_1_ratio_dist['lt_2_5pct']} 个 ({test_1_ratio_dist['lt_2_5pct']/test_1_total*100:.2f}%)")
print(f"小于5%: {test_1_ratio_dist['lt_5pct']} 个 ({test_1_ratio_dist['lt_5pct']/test_1_total*100:.2f}%)")
print(f"小于10%: {test_1_ratio_dist['lt_10pct']} 个 ({test_1_ratio_dist['lt_10pct']/test_1_total*100:.2f}%)")
print(f"小于15%: {test_1_ratio_dist['lt_15pct']} 个 ({test_1_ratio_dist['lt_15pct']/test_1_total*100:.2f}%)")
print(f"小于20%: {test_1_ratio_dist['lt_20pct']} 个 ({test_1_ratio_dist['lt_20pct']/test_1_total*100:.2f}%)")
print(f"大于等于20%: {test_1_ratio_dist['gt_20pct']} 个 ({test_1_ratio_dist['gt_20pct']/test_1_total*100:.2f}%)")
print(f"{'='*80}")