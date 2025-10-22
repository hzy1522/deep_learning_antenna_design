import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import random

# 设置随机种子以确保可重复性
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


class RobustAntennaDataset(data.Dataset):
    """
    鲁棒版天线数据集类
    改进数据生成和增强策略
    """

    def __init__(self, num_samples=15000, normalize=True, augment=True):
        self.base_num_samples = num_samples
        self.normalize = normalize
        self.augment = augment

        # 生成天线参数数据
        self.generate_antenna_data()

        if normalize:
            self.normalize_data()

    def generate_antenna_data(self):
        """生成更真实、更多样的天线设计数据"""
        # 基础参数
        self.parameters = {
            # 贴片参数
            'patch_length': np.random.uniform(8, 45, self.base_num_samples),  # 缩小范围，更聚焦
            'patch_width': np.random.uniform(5, 25, self.base_num_samples),  # 缩小范围

            # 介质板参数
            'substrate_thickness': np.random.uniform(0.5, 4, self.base_num_samples),  # 更合理的厚度范围
            'substrate_epsr': np.random.uniform(2.5, 8.0, self.base_num_samples),  # 常用介质范围
            'substrate_length': None,
            'substrate_width': None,

            # 接地平面参数
            'ground_length': None,
            'ground_width': None,
            'ground_thickness': np.random.choice([0.017, 0.035, 0.070], self.base_num_samples),  # 标准厚度

            # 工作参数
            'operating_frequency': np.random.uniform(1.0, 12, self.base_num_samples),  # 常用频率范围
            'feed_position': np.random.uniform(0.2, 0.8, self.base_num_samples)  # 更合理的馈电位置
        }

        # 计算相关参数（添加更多变化）
        substrate_extension = np.random.uniform(1.1, 1.8, self.base_num_samples)
        self.parameters['substrate_length'] = self.parameters['patch_length'] * substrate_extension
        self.parameters['substrate_width'] = self.parameters['patch_width'] * substrate_extension

        ground_variation = np.random.uniform(0.9, 1.1, self.base_num_samples)
        self.parameters['ground_length'] = self.parameters['substrate_length'] * ground_variation
        self.parameters['ground_width'] = self.parameters['substrate_width'] * ground_variation

        # 基于改进的电磁理论计算性能指标
        self.performances = self.calculate_performances()

        # 数据增强
        if self.augment and self.base_num_samples > 5000:
            self.augment_data()

    def calculate_performances(self, params=None):
        """改进的性能计算模型"""
        if params is None:
            params = self.parameters

        c = 3e8  # 光速

        # 提取参数
        patch_length = params['patch_length']
        patch_width = params['patch_width']
        substrate_thickness = params['substrate_thickness']
        substrate_epsr = params['substrate_epsr']
        operating_freq = params['operating_frequency']
        substrate_length = params['substrate_length']
        substrate_width = params['substrate_width']
        ground_length = params['ground_length']
        ground_width = params['ground_width']
        feed_position = params['feed_position']

        # 确保参数有效
        patch_width = np.maximum(patch_width, 1e-3)
        substrate_thickness = np.maximum(substrate_thickness, 1e-3)
        substrate_epsr = np.maximum(substrate_epsr, 1.0)

        # 1. 计算谐振频率
        effective_epsr = (substrate_epsr + 1) / 2 + (substrate_epsr - 1) / 2 * \
                         np.power(1 + 12 * substrate_thickness / patch_width, -0.5)

        delta_l = substrate_thickness * (0.412 * (effective_epsr + 0.3) * (patch_width / substrate_thickness + 0.264) /
                                         ((effective_epsr - 0.258) * (patch_width / substrate_thickness + 0.8)))

        effective_length = patch_length + 2 * delta_l
        effective_length = np.maximum(effective_length, 1e-3)

        resonance_frequency = (c / (2 * effective_length * 1e-3 * np.sqrt(effective_epsr))) / 1e9
        resonance_frequency = np.maximum(resonance_frequency, 0.1)  # 确保频率合理

        # 2. 计算带宽
        Q_radiation = (np.pi * np.sqrt(effective_epsr) * patch_width) / (2 * substrate_thickness)
        Q_dielectric = 1 / (np.tan(np.pi * substrate_epsr * substrate_thickness * operating_freq * 1e9 * 2 * np.pi / c))
        Q_conductor = (np.pi * np.sqrt(effective_epsr) * patch_width * np.sqrt(operating_freq * 1e9)) / (
                    2 * substrate_thickness * 6.62e4)

        # 确保Q值有效
        Q_radiation = np.maximum(Q_radiation, 1)
        Q_dielectric = np.maximum(Q_dielectric, 1)
        Q_conductor = np.maximum(Q_conductor, 1)

        Q_total = 1 / (1 / Q_radiation + 1 / Q_dielectric + 1 / Q_conductor)
        fractional_bandwidth = 1.8 / Q_total
        bandwidth = fractional_bandwidth * resonance_frequency * 1000

        # 介质板尺寸影响
        substrate_aspect_ratio = substrate_length / substrate_width
        bandwidth *= (0.85 + 0.15 * substrate_aspect_ratio)
        bandwidth = np.clip(bandwidth, 10, 400)

        # 3. 计算增益
        radiation_efficiency = Q_radiation / Q_total
        radiation_efficiency = np.clip(radiation_efficiency, 0.01, 0.99)

        wavelength = c / (resonance_frequency * 1e9)
        directivity = (4 * np.pi * (patch_length * 1e-3) * (patch_width * 1e-3)) / (wavelength ** 2)
        directivity = np.maximum(directivity, 0.1)

        directivity_efficiency_product = directivity * radiation_efficiency
        gain = 10 * np.log10(directivity_efficiency_product)

        # 接地平面影响
        ground_plane_factor = np.minimum(1.0, (ground_length * ground_width) / (patch_length * patch_width * 4))
        gain *= (0.6 + 0.4 * ground_plane_factor)

        # 介质板厚度影响
        thickness_factor = np.exp(-0.08 * substrate_thickness)
        gain *= (0.85 + 0.15 * thickness_factor)

        gain = np.clip(gain, -5, 15)

        # 4. 计算回波损耗 S11
        Z0 = 377 / np.sqrt(effective_epsr)
        Z_patch = (120 * np.pi ** 2) / (120 * effective_epsr * (patch_width / patch_length) + 30 * np.pi)

        # 确保阻抗有效
        Z_patch = np.maximum(Z_patch, 10)
        Z_patch = np.minimum(Z_patch, 200)

        feed_offset = np.abs(feed_position - 0.5)
        impedance_mismatch = Z_patch * (1 + 0.6 * feed_offset)

        reflection_coefficient = (impedance_mismatch - 50) / (impedance_mismatch + 50)
        s11 = 20 * np.log10(np.abs(reflection_coefficient))

        # 频率失配影响
        freq_mismatch = np.abs(resonance_frequency - operating_freq) / resonance_frequency
        s11 += 10 * freq_mismatch * 15  # 调整系数，使结果更合理

        s11 = np.clip(s11, -35, -3)

        # 添加适度噪声
        noise_level = 0.05  # 降低噪声水平
        num_samples = len(patch_length)
        bandwidth += np.random.normal(0, 8 * noise_level, num_samples)
        gain += np.random.normal(0, 0.6 * noise_level, num_samples)
        s11 += np.random.normal(0, 1.5 * noise_level, num_samples)

        return {
            'resonance_frequency': resonance_frequency,
            'bandwidth': bandwidth,
            'gain': gain,
            's11': s11
        }

    def augment_data(self):
        """数据增强：增加样本多样性"""
        print("正在进行数据增强...")

        # 对10%的数据进行轻微扰动
        augment_ratio = 0.1
        augment_size = int(self.base_num_samples * augment_ratio)

        # 随机选择要增强的样本
        indices = np.random.choice(self.base_num_samples, augment_size, replace=False)

        # 创建增强参数
        augmented_params = {}
        for key in self.parameters:
            if key not in ['ground_thickness']:  # 不对接地厚度进行增强
                # 轻微扰动：±5%
                perturbation = np.random.uniform(0.95, 1.05, augment_size)
                augmented_params[key] = self.parameters[key][indices] * perturbation
            else:
                augmented_params[key] = self.parameters[key][indices]

        # 计算增强数据的性能指标
        augmented_performances = self.calculate_performances(augmented_params)

        # 合并原始数据和增强数据
        for key in self.parameters:
            self.parameters[key] = np.concatenate([
                self.parameters[key],
                augmented_params[key]
            ])

        for key in self.performances:
            self.performances[key] = np.concatenate([
                self.performances[key],
                augmented_performances[key]
            ])

        self.base_num_samples += augment_size
        print(f"数据增强完成，总样本数：{self.base_num_samples}")

    def normalize_data(self):
        """数据归一化"""
        self.param_stats = {}
        self.perf_stats = {}

        # 归一化参数
        for key in self.parameters:
            mean = np.mean(self.parameters[key])
            std = np.std(self.parameters[key])
            self.param_stats[key] = (mean, std)
            self.parameters[key] = (self.parameters[key] - mean) / std

        # 归一化性能指标
        for key in self.performances:
            mean = np.mean(self.performances[key])
            std = np.std(self.performances[key])
            self.perf_stats[key] = (mean, std)
            self.performances[key] = (self.performances[key] - mean) / std

    def __len__(self):
        return self.base_num_samples

    def __getitem__(self, idx):
        """获取数据项"""
        param_values = np.array([self.parameters[key][idx] for key in self.parameters.keys()], dtype=np.float32)
        perf_values = np.array([self.performances[key][idx] for key in self.performances.keys()], dtype=np.float32)

        return torch.tensor(param_values), torch.tensor(perf_values)


class RobustAntennaForwardModel(nn.Module):
    """
    鲁棒版正向预测模型
    增加正则化，减少过拟合
    """

    def __init__(self, input_dim=11, output_dim=4, dropout_rate=0.3):
        super(RobustAntennaForwardModel, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 1.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class RobustAntennaInverseModel(nn.Module):
    """
    鲁棒版逆向设计模型
    """

    def __init__(self, input_dim=4, output_dim=11, dropout_rate=0.3):
        super(RobustAntennaInverseModel, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 1.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.network(x)


class EarlyStopping:
    """早停机制：防止过拟合"""

    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """保存最佳模型"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class ImprovedAntennaDesignFramework:
    """
    改进版天线设计框架
    增加抗过拟合机制
    """

    def __init__(self, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 数据集
        self.dataset = None
        self.train_loader = None
        self.test_loader = None

        # 模型
        self.forward_model = None
        self.inverse_model = None

        # 训练历史
        self.forward_history = {'loss': [], 'val_loss': []}
        self.inverse_history = {'loss': [], 'val_loss': []}

    def prepare_data(self, num_samples=15000, test_ratio=0.2, batch_size=64):
        """准备训练和测试数据"""
        print("正在生成鲁棒版天线数据集...")
        self.dataset = RobustAntennaDataset(num_samples=num_samples, augment=True)

        # 获取实际数据集大小（可能包含增强数据）
        actual_dataset_size = len(self.dataset)
        test_size = int(actual_dataset_size * test_ratio)
        train_size = actual_dataset_size - test_size

        print(f"实际数据集大小: {actual_dataset_size} (包含增强数据)")
        print(f"划分: 训练集 {train_size} 样本, 测试集 {test_size} 样本")

        train_dataset, test_dataset = data.random_split(
            self.dataset, [train_size, test_size]
        )

        self.train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"数据集准备完成: 训练集 {train_size} 样本, 测试集 {test_size} 样本")
        print(f"输入参数数量: {len(self.dataset.parameters)}")

    def build_models(self, dropout_rate=0.3):
        """构建鲁棒版模型"""
        print("正在构建鲁棒版神经网络模型...")

        # 正向预测模型
        self.forward_model = RobustAntennaForwardModel(
            input_dim=len(self.dataset.parameters),
            output_dim=4,
            dropout_rate=dropout_rate
        ).to(self.device)

        # 逆向设计模型
        self.inverse_model = RobustAntennaInverseModel(
            input_dim=4,
            output_dim=len(self.dataset.parameters),
            dropout_rate=dropout_rate
        ).to(self.device)

        print("模型构建完成")

    def train_forward_model(self, epochs=50, lr=0.001, patience=10):
        """训练正向预测模型（带早停机制）"""
        print("\n开始训练鲁棒版正向预测模型...")

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.forward_model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        # 早停机制
        early_stopping = EarlyStopping(patience=patience, verbose=True, path='./robust_earlystopping/best_forward_model.pt')

        for epoch in range(epochs):
            self.forward_model.train()
            total_loss = 0

            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # 前向传播
                outputs = self.forward_model(inputs)
                loss = criterion(outputs, targets)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.forward_model.parameters(), max_norm=1.0)  # 梯度裁剪
                optimizer.step()

                total_loss += loss.item() * inputs.size(0)

            # 计算平均损失
            avg_loss = total_loss / len(self.train_loader.dataset)
            self.forward_history['loss'].append(avg_loss)

            # 验证
            val_loss = self.evaluate_model(self.forward_model, self.test_loader)
            self.forward_history['val_loss'].append(val_loss)

            # 学习率调度
            scheduler.step(val_loss)

            # 早停检查
            early_stopping(val_loss, self.forward_model)

            if (epoch + 1) % 5 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f"正向模型 - 第 {epoch + 1}/{epochs} 轮: 训练损失 {avg_loss:.6f}, 验证损失 {val_loss:.6f}, 学习率 {current_lr:.6f}")

            if early_stopping.early_stop:
                print("早停机制触发，停止训练")
                break

        # 加载最佳模型
        self.forward_model.load_state_dict(torch.load('./robust_earlystopping/best_forward_model.pt'))
        print("正向预测模型训练完成")

    def train_inverse_model(self, epochs=50, lr=0.001, patience=10):
        """训练逆向设计模型"""
        print("\n开始训练鲁棒版逆向设计模型...")

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.inverse_model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        # 早停机制
        early_stopping = EarlyStopping(patience=patience, verbose=True, path='./robust_earlystopping/best_inverse_model.pt')

        for epoch in range(epochs):
            self.inverse_model.train()
            total_loss = 0

            for params, performances in self.train_loader:
                performances, params = performances.to(self.device), params.to(self.device)

                # 前向传播
                outputs = self.inverse_model(performances)
                loss = criterion(outputs, params)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.inverse_model.parameters(), max_norm=1.0)  # 梯度裁剪
                optimizer.step()

                total_loss += loss.item() * performances.size(0)

            # 计算平均损失
            avg_loss = total_loss / len(self.train_loader.dataset)
            self.inverse_history['loss'].append(avg_loss)

            # 验证
            val_loss = self.evaluate_inverse_model(self.inverse_model, self.test_loader)
            self.inverse_history['val_loss'].append(val_loss)

            # 学习率调度
            scheduler.step(val_loss)

            # 早停检查
            early_stopping(val_loss, self.inverse_model)

            if (epoch + 1) % 5 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f"逆向模型 - 第 {epoch + 1}/{epochs} 轮: 训练损失 {avg_loss:.6f}, 验证损失 {val_loss:.6f}, 学习率 {current_lr:.6f}")

            if early_stopping.early_stop:
                print("早停机制触发，停止训练")
                break

        # 加载最佳模型
        self.inverse_model.load_state_dict(torch.load('./robust_earlystopping/best_inverse_model.pt'))
        print("逆向设计模型训练完成")

    def evaluate_model(self, model, dataloader):
        """评估模型性能"""
        model.eval()
        criterion = nn.MSELoss()
        total_loss = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)

        return total_loss / len(dataloader.dataset)

    def evaluate_inverse_model(self, model, dataloader):
        """评估逆向模型性能"""
        model.eval()
        criterion = nn.MSELoss()
        total_loss = 0

        with torch.no_grad():
            for params, performances in dataloader:
                performances, params = performances.to(self.device), params.to(self.device)
                outputs = model(performances)
                loss = criterion(outputs, params)
                total_loss += loss.item() * performances.size(0)

        return total_loss / len(dataloader.dataset)

    def predict_performance(self, parameters):
        """预测天线性能"""
        self.forward_model.eval()

        normalized_params = []
        param_keys = list(self.dataset.parameters.keys())
        for key in param_keys:
            if key in parameters:
                mean, std = self.dataset.param_stats[key]
                normalized_params.append((parameters[key] - mean) / std)
            else:
                mean, _ = self.dataset.param_stats[key]
                normalized_params.append((mean - mean) / 1.0)

        input_tensor = torch.tensor(normalized_params, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            normalized_output = self.forward_model(input_tensor)

        output_dict = {}
        perf_keys = ['resonance_frequency', 'bandwidth', 'gain', 's11']
        for i, key in enumerate(perf_keys):
            mean, std = self.dataset.perf_stats[key]
            output_dict[key] = normalized_output[0, i].item() * std + mean

        return output_dict

    def design_antenna(self, target_performances):
        """基于目标性能设计天线结构"""
        self.inverse_model.eval()

        normalized_targets = []
        perf_keys = ['resonance_frequency', 'bandwidth', 'gain', 's11']
        for key in perf_keys:
            if key in target_performances:
                mean, std = self.dataset.perf_stats[key]
                normalized_targets.append((target_performances[key] - mean) / std)
            else:
                mean, _ = self.dataset.perf_stats[key]
                normalized_targets.append((mean - mean) / 1.0)

        input_tensor = torch.tensor(normalized_targets, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            normalized_output = self.inverse_model(input_tensor)

        output_dict = {}
        param_keys = list(self.dataset.parameters.keys())
        for i, key in enumerate(param_keys):
            mean, std = self.dataset.param_stats[key]
            output_dict[key] = normalized_output[0, i].item() * std + mean

        return output_dict

    def visualize_training_history(self, save_path='./robust_picture/robust_training_history.png'):
        """可视化训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 正向模型训练历史
        ax1.plot(self.forward_history['loss'], label='训练损失')
        ax1.plot(self.forward_history['val_loss'], label='验证损失')
        ax1.set_title('鲁棒版正向预测模型训练历史')
        ax1.set_xlabel('轮次')
        ax1.set_ylabel('损失')
        ax1.legend()
        ax1.grid(True)

        # 计算并显示最终损失
        final_train_loss = self.forward_history['loss'][-1]
        final_val_loss = self.forward_history['val_loss'][-1]
        ax1.text(0.02, 0.98, f'最终训练损失: {final_train_loss:.4f}\n最终验证损失: {final_val_loss:.4f}',
                 transform=ax1.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 逆向模型训练历史
        ax2.plot(self.inverse_history['loss'], label='训练损失')
        ax2.plot(self.inverse_history['val_loss'], label='验证损失')
        ax2.set_title('鲁棒版逆向设计模型训练历史')
        ax2.set_xlabel('轮次')
        ax2.set_ylabel('损失')
        ax2.legend()
        ax2.grid(True)

        if self.inverse_history['loss']:
            final_train_loss_inv = self.inverse_history['loss'][-1]
            final_val_loss_inv = self.inverse_history['val_loss'][-1]
            ax2.text(0.02, 0.98, f'最终训练损失: {final_train_loss_inv:.4f}\n最终验证损失: {final_val_loss_inv:.4f}',
                     transform=ax2.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"训练历史图已保存为 {save_path}")

    def visualize_antenna_structure(self, parameters, save_path='./robust_picture/robust_antenna_design.png'):
        """可视化天线结构设计"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 俯视图
        patch_length = parameters['patch_length']
        patch_width = parameters['patch_width']
        substrate_length = parameters['substrate_length']
        substrate_width = parameters['substrate_width']
        ground_length = parameters['ground_length']
        ground_width = parameters['ground_width']
        feed_position = parameters['feed_position']

        # 绘制接地平面
        ground = patches.Rectangle((-ground_length / 2, -ground_width / 2),
                                   ground_length, ground_width,
                                   linewidth=1, edgecolor='gray', facecolor='lightgray', alpha=0.5)
        ax1.add_patch(ground)

        # 绘制介质板
        substrate = patches.Rectangle((-substrate_length / 2, -substrate_width / 2),
                                      substrate_length, substrate_width,
                                      linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.6)
        ax1.add_patch(substrate)

        # 绘制贴片
        patch = patches.Rectangle((-patch_length / 2, -patch_width / 2),
                                  patch_length, patch_width,
                                  linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.8)
        ax1.add_patch(patch)

        # 绘制馈电点
        feed_x = -patch_length / 2 + patch_length * feed_position
        feed_point = patches.Circle((feed_x, 0), 0.5, color='red')
        ax1.add_patch(feed_point)

        ax1.set_xlim(-ground_length / 2 - 2, ground_length / 2 + 2)
        ax1.set_ylim(-ground_width / 2 - 2, ground_width / 2 + 2)
        ax1.set_aspect('equal')
        ax1.set_title('天线俯视图')
        ax1.grid(True, alpha=0.3)

        # 侧视图
        substrate_thickness = parameters['substrate_thickness']
        ground_thickness = parameters['ground_thickness']

        ground_side = patches.Rectangle((-ground_length / 2, -ground_thickness),
                                        ground_length, ground_thickness,
                                        linewidth=1, edgecolor='gray', facecolor='lightgray', alpha=0.5)
        ax2.add_patch(ground_side)

        substrate_side = patches.Rectangle((-substrate_length / 2, 0),
                                           substrate_length, substrate_thickness,
                                           linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.6)
        ax2.add_patch(substrate_side)

        patch_side = patches.Rectangle((-patch_length / 2, substrate_thickness),
                                       patch_length, 0.1,
                                       linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.8)
        ax2.add_patch(patch_side)

        ax2.set_xlim(-ground_length / 2 - 2, ground_length / 2 + 2)
        ax2.set_ylim(-ground_thickness - 1, substrate_thickness + 1)
        ax2.set_aspect('equal')
        ax2.set_title('天线侧视图')
        ax2.grid(True, alpha=0.3)

        # 参数表格
        ax3.axis('tight')
        ax3.axis('off')

        param_data = [
            ['参数名称', '数值', '单位'],
            ['贴片长度', f"{patch_length:.2f}", 'mm'],
            ['贴片宽度', f"{patch_width:.2f}", 'mm'],
            ['介质板厚度', f"{substrate_thickness:.2f}", 'mm'],
            ['介质常数', f"{parameters['substrate_epsr']:.2f}", ''],
            ['工作频率', f"{parameters['operating_frequency']:.2f}", 'GHz'],
            ['馈电位置', f"{feed_position:.2f}", '（相对长度）'],
        ]

        table = ax3.table(cellText=param_data[1:], colLabels=param_data[0],
                          cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax3.set_title('天线参数', fontsize=14)

        # 性能指标图表
        if 'resonance_frequency' in parameters:
            performance_data = [
                ['谐振频率', f"{parameters['resonance_frequency']:.2f}", 'GHz'],
                ['带宽', f"{parameters['bandwidth']:.2f}", 'MHz'],
                ['增益', f"{parameters['gain']:.2f}", 'dBi'],
                ['回波损耗', f"{parameters['s11']:.2f}", 'dB'],
            ]

            ax4.axis('tight')
            ax4.axis('off')
            perf_table = ax4.table(cellText=performance_data,
                                   colLabels=['性能指标', '数值', '单位'],
                                   cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
            perf_table.auto_set_font_size(False)
            perf_table.set_fontsize(10)
            perf_table.scale(1, 2)
            ax4.set_title('预期性能', fontsize=14)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"天线结构图已保存为 {save_path}")

    def save_models(self, save_dir='robust_antenna_models'):
        """保存训练好的模型"""
        os.makedirs(save_dir, exist_ok=True)

        # 保存正向模型
        forward_path = os.path.join(save_dir, 'robust_forward_model.pth')
        torch.save(self.forward_model.state_dict(), forward_path)

        # 保存逆向模型
        inverse_path = os.path.join(save_dir, 'robust_inverse_model.pth')
        torch.save(self.inverse_model.state_dict(), inverse_path)

        # 保存数据集统计信息
        stats_path = os.path.join(save_dir, 'robust_data_stats.npy')
        stats = {
            'param_stats': self.dataset.param_stats,
            'perf_stats': self.dataset.perf_stats,
            'param_names': list(self.dataset.parameters.keys()),
            'perf_names': ['resonance_frequency', 'bandwidth', 'gain', 's11']
        }
        np.save(stats_path, stats)

        print(f"鲁棒版模型已保存到 {save_dir} 目录")

    def load_models(self, load_dir='robust_antenna_models'):
        """加载训练好的模型"""
        import os
        import numpy as np

        # 加载统计信息
        stats_path = os.path.join(load_dir, 'robust_data_stats.npy')
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"统计信息文件不存在: {stats_path}")

        stats = np.load(stats_path, allow_pickle=True).item()

        # 创建虚拟数据集用于反归一化
        self.dataset = RobustAntennaDataset(num_samples=1, augment=False)
        self.dataset.param_stats = stats['param_stats']
        self.dataset.perf_stats = stats['perf_stats']
        self.dataset.parameters = {key: np.zeros(1) for key in stats['param_names']}
        self.dataset.performances = {key: np.zeros(1) for key in stats['perf_names']}

        # 加载正向模型
        forward_path = os.path.join(load_dir, 'robust_forward_model.pth')
        if not os.path.exists(forward_path):
            raise FileNotFoundError(f"正向模型文件不存在: {forward_path}")

        self.forward_model = RobustAntennaForwardModel(
            input_dim=len(stats['param_names']),
            output_dim=len(stats['perf_names'])
        ).to(self.device)
        self.forward_model.load_state_dict(torch.load(forward_path, map_location=self.device))
        self.forward_model.eval()

        # 加载逆向模型
        inverse_path = os.path.join(load_dir, 'robust_inverse_model.pth')
        if not os.path.exists(inverse_path):
            raise FileNotFoundError(f"逆向模型文件不存在: {inverse_path}")

        self.inverse_model = RobustAntennaInverseModel(
            input_dim=len(stats['perf_names']),
            output_dim=len(stats['param_names'])
        ).to(self.device)
        self.inverse_model.load_state_dict(torch.load(inverse_path, map_location=self.device))
        self.inverse_model.eval()

        print(f"鲁棒版模型已从 {load_dir} 目录加载")

def main():
    """主函数"""
    print("=" * 60)
    print("鲁棒版基于PyTorch的深度学习天线结构设计框架")
    print("版本: 2.1 - 解决过拟合问题")
    print("=" * 60)

    # 创建改进版天线设计框架实例
    framework = ImprovedAntennaDesignFramework()

    # 步骤1: 准备数据
    framework.prepare_data(num_samples=15000, batch_size=64)

    # 步骤2: 构建模型
    framework.build_models(dropout_rate=0.25)

    # 步骤3: 训练模型（带早停机制）
    framework.train_forward_model(epochs=50, lr=0.001, patience=8)
    framework.train_inverse_model(epochs=50, lr=0.001, patience=8)

    # 步骤4: 可视化训练历史
    framework.visualize_training_history()

    # 步骤5: 保存模型
    framework.save_models()

    print("\n" + "=" * 60)
    print("鲁棒版模型训练完成！开始演示天线设计功能...")
    print("=" * 60)

    # 演示1: 正向预测
    print("\n【演示1: 正向预测】")
    test_parameters = {
        'patch_length': 20.0,  # mm
        'patch_width': 15.0,  # mm
        'substrate_thickness': 1.6,  # mm
        'substrate_epsr': 4.4,  # FR-4材料
        'substrate_length': 30.0,  # mm
        'substrate_width': 25.0,  # mm
        'ground_length': 30.0,  # mm
        'ground_width': 25.0,  # mm
        'ground_thickness': 0.035,  # mm
        'operating_frequency': 2.4,  # GHz
        'feed_position': 0.25  # 相对位置
    }

    predicted_performance = framework.predict_performance(test_parameters)
    print(f"输入参数: {test_parameters}")
    print(f"预测性能: {predicted_performance}")

    # 演示2: 逆向设计
    print("\n【演示2: 逆向设计】")
    target_performances = {
        'resonance_frequency': 2.4,  # 目标谐振频率 2.4GHz
        'bandwidth': 80,  # 目标带宽 80MHz
        'gain': 5.0,  # 目标增益 5dBi
        's11': -18.0  # 目标回波损耗 -18dB
    }

    designed_parameters = framework.design_antenna(target_performances)
    print(f"目标性能: {target_performances}")
    print(f"设计参数: {designed_parameters}")

    # 验证设计结果
    verified_performance = framework.predict_performance(designed_parameters)
    print(f"验证性能: {verified_performance}")

    # 可视化设计结果
    design_with_performance = {**designed_parameters, **verified_performance}
    framework.visualize_antenna_structure(design_with_performance, './robust_picture/robust_designed_antenna.png')

    print("\n" + "=" * 60)
    print("鲁棒版天线设计演示完成！")
    print("生成的文件:")
    print("- robust_training_history.png: 训练历史图")
    print("- robust_designed_antenna.png: 天线设计结构图")
    print("- robust_antenna_models/: 鲁棒版训练模型文件")
    print("=" * 60)

    # 输出改进总结
    print("\n【改进总结】")
    print("1. 数据改进:")
    print("   - 增加样本数量到15000个")
    print("   - 缩小参数范围，提高数据质量")
    print("   - 增加数据增强机制")
    print("   - 降低噪声水平，提高数据可靠性")
    print("   - 增加数值稳定性检查")
    print("\n2. 模型改进:")
    print("   - 简化网络结构，减少过拟合风险")
    print("   - 增加Dropout层，提高泛化能力")
    print("   - 使用AdamW优化器，增加权重衰减")
    print("   - 添加梯度裁剪，防止梯度爆炸")
    print("\n3. 训练策略改进:")
    print("   - 增加早停机制，及时停止过拟合")
    print("   - 使用学习率调度，自动调整学习率")
    print("   - 减少训练轮次，防止过度训练")
    print("   - 增加验证频率，更好监控训练状态")


if __name__ == "__main__":
    main()