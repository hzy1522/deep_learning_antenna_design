"""
改进版天线设计框架 - 考虑介质板和接地部分影响
Enhanced Antenna Design Framework - Considering Substrate and Ground Plane Effects

作者: 豆包AI助手
日期: 2025年10月22日
版本: 2.0
"""

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


class EnhancedAntennaDataset(data.Dataset):
    """
    改进版天线数据集类
    考虑介质板和接地部分的详细参数
    """

    def __init__(self, num_samples=10000, normalize=True):
        self.num_samples = num_samples
        self.normalize = normalize

        # 生成天线参数数据（包含更多详细参数）
        self.generate_antenna_data()

        if normalize:
            self.normalize_data()

    def generate_antenna_data(self):
        """生成包含详细参数的天线设计数据"""
        # 基础参数
        self.parameters = {
            # 贴片参数
            'patch_length': np.random.uniform(5, 50, self.num_samples),  # 贴片长度 5-50mm
            'patch_width': np.random.uniform(2, 30, self.num_samples),  # 贴片宽度 2-30mm

            # 介质板参数
            'substrate_thickness': np.random.uniform(0.2, 5, self.num_samples),  # 介质板厚度 0.2-5mm
            'substrate_epsr': np.random.uniform(2.2, 10.2, self.num_samples),  # 介质板介电常数
            'substrate_length': None,  # 介质板长度（由贴片长度计算）
            'substrate_width': None,  # 介质板宽度（由贴片宽度计算）

            # 接地平面参数
            'ground_length': None,  # 接地平面长度
            'ground_width': None,  # 接地平面宽度
            'ground_thickness': np.random.uniform(0.017, 0.070, self.num_samples),  # 接地平面厚度 0.017-0.070mm

            # 工作参数
            'operating_frequency': np.random.uniform(0.5, 15, self.num_samples),  # 工作频率 0.5-15GHz
            'feed_position': np.random.uniform(0.1, 0.9, self.num_samples)  # 馈电位置 0.1-0.9（相对长度）
        }

        # 计算相关参数
        self.parameters['substrate_length'] = self.parameters['patch_length'] * np.random.uniform(1.2, 2.0,
                                                                                                  self.num_samples)
        self.parameters['substrate_width'] = self.parameters['patch_width'] * np.random.uniform(1.2, 2.0,
                                                                                                self.num_samples)
        self.parameters['ground_length'] = self.parameters['substrate_length'] * np.random.uniform(0.8, 1.2,
                                                                                                   self.num_samples)
        self.parameters['ground_width'] = self.parameters['substrate_width'] * np.random.uniform(0.8, 1.2,
                                                                                                 self.num_samples)

        # 基于改进的电磁理论计算性能指标
        self.performances = self.calculate_performances_improved()

    def calculate_performances_improved(self):
        """
        改进版天线性能计算模型
        考虑介质板和接地部分的详细参数影响
        """
        params = self.parameters
        c = 3e8  # 光速

        # 提取参数
        patch_length = params['patch_length']  # 贴片长度 (mm)
        patch_width = params['patch_width']  # 贴片宽度 (mm)
        substrate_thickness = params['substrate_thickness']  # 介质板厚度 (mm)
        substrate_epsr = params['substrate_epsr']  # 介质板介电常数
        operating_freq = params['operating_frequency']  # 工作频率 (GHz)
        substrate_length = params['substrate_length']  # 介质板长度 (mm)
        substrate_width = params['substrate_width']  # 介质板宽度 (mm)
        ground_length = params['ground_length']  # 接地平面长度 (mm)
        ground_width = params['ground_width']  # 接地平面宽度 (mm)
        ground_thickness = params['ground_thickness']  # 接地平面厚度 (mm)
        feed_position = params['feed_position']  # 馈电位置

        # 1. 计算谐振频率（考虑边缘场效应和介质板尺寸）
        # 有效介电常数
        effective_epsr = (substrate_epsr + 1) / 2 + (substrate_epsr - 1) / 2 * \
                         np.power(1 + 12 * substrate_thickness / patch_width, -0.5)

        # 边缘场扩展长度
        delta_l = substrate_thickness * (0.412 * (effective_epsr + 0.3) * (patch_width / substrate_thickness + 0.264) /
                                         ((effective_epsr - 0.258) * (patch_width / substrate_thickness + 0.8)))

        # 有效贴片长度
        effective_length = patch_length + 2 * delta_l

        # 谐振频率
        resonance_frequency = (c / (2 * effective_length * 1e-3 * np.sqrt(effective_epsr))) / 1e9

        # 2. 计算带宽（考虑介质板和接地影响）
        # 品质因数 Q
        Q_radiation = (np.pi * np.sqrt(effective_epsr) * patch_width) / (2 * substrate_thickness)
        Q_dielectric = 1 / (np.tan(np.pi * substrate_epsr * substrate_thickness * operating_freq * 1e9 * 2 * np.pi / c))
        Q_conductor = (np.pi * np.sqrt(effective_epsr) * patch_width * np.sqrt(operating_freq * 1e9)) / (
                    2 * substrate_thickness * 6.62e4)

        Q_total = 1 / (1 / Q_radiation + 1 / Q_dielectric + 1 / Q_conductor)

        # 相对带宽
        fractional_bandwidth = 1.8 / Q_total

        # 绝对带宽 (MHz)
        bandwidth = fractional_bandwidth * resonance_frequency * 1000

        # 考虑介质板尺寸影响的带宽修正
        substrate_aspect_ratio = substrate_length / substrate_width
        bandwidth *= (0.8 + 0.2 * substrate_aspect_ratio)  # 长宽比修正

        bandwidth = np.clip(bandwidth, 5, 500)  # 限制在合理范围内

        # 3. 计算增益（考虑接地平面和介质影响）
        # 天线效率
        radiation_efficiency = Q_radiation / Q_total

        # 方向性系数
        wavelength = c / (resonance_frequency * 1e9)
        directivity = (4 * np.pi * (patch_length * 1e-3) * (patch_width * 1e-3)) / (wavelength ** 2)

        # 确保输入为正数
        directivity_efficiency_product = np.maximum(directivity * radiation_efficiency, 1e-10)

        # 增益 = 方向性 × 效率
        gain = 10 * np.log10(directivity_efficiency_product)

        # 接地平面尺寸影响
        ground_plane_factor = np.minimum(1.0, (ground_length * ground_width) / (patch_length * patch_width * 4))
        gain *= (0.5 + 0.5 * ground_plane_factor)  # 接地平面越大，增益越高

        # 介质板厚度影响
        thickness_factor = np.exp(-0.1 * substrate_thickness)
        gain *= (0.8 + 0.2 * thickness_factor)

        gain = np.clip(gain, -10, 20)  # 限制在合理范围内

        # 4. 计算回波损耗 S11（考虑馈电位置和阻抗匹配）
        # 输入阻抗计算
        Z0 = 377 / np.sqrt(effective_epsr)  # 介质中的波阻抗
        Z_patch = (120 * np.pi ** 2) / (120 * effective_epsr * (patch_width / patch_length) + 30 * np.pi)

        # 馈电位置对阻抗的影响
        feed_offset = np.abs(feed_position - 0.5)  # 偏离中心的程度
        impedance_mismatch = Z_patch * (1 + 0.8 * feed_offset)

        # 反射系数
        reflection_coefficient = (impedance_mismatch - 50) / (impedance_mismatch + 50)
        s11 = 20 * np.log10(np.abs(reflection_coefficient))

        # 工作频率与谐振频率失配影响
        freq_mismatch = np.abs(resonance_frequency - operating_freq) / resonance_frequency
        s11 += 10 * freq_mismatch * 20  # 频率失配导致S11恶化

        s11 = np.clip(s11, -40, 0)  # 限制在合理范围内

        # 添加一些随机噪声模拟制造和环境变化
        bandwidth += np.random.normal(0, 10, self.num_samples)
        gain += np.random.normal(0, 0.8, self.num_samples)
        s11 += np.random.normal(0, 2, self.num_samples)

        return {
            'resonance_frequency': resonance_frequency,
            'bandwidth': bandwidth,
            'gain': gain,
            's11': s11,
            'effective_epsr': effective_epsr,
            'radiation_efficiency': radiation_efficiency
        }

    def normalize_data(self):
        """数据归一化"""
        # 保存原始统计信息
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

    def denormalize_parameters(self, normalized_params):
        """反归一化参数"""
        denorm_params = {}
        for i, key in enumerate(self.parameters.keys()):
            mean, std = self.param_stats[key]
            denorm_params[key] = normalized_params[:, i] * std + mean
        return denorm_params

    def denormalize_performances(self, normalized_perfs):
        """反归一化性能指标"""
        denorm_perfs = {}
        for i, key in enumerate(self.performances.keys()):
            mean, std = self.perf_stats[key]
            denorm_perfs[key] = normalized_perfs[:, i] * std + mean
        return denorm_perfs

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """获取数据项"""
        param_values = np.array([self.parameters[key][idx] for key in self.parameters.keys()], dtype=np.float32)
        perf_values = np.array([self.performances[key][idx] for key in self.performances.keys() if
                                key in ['resonance_frequency', 'bandwidth', 'gain', 's11']], dtype=np.float32)

        return torch.tensor(param_values), torch.tensor(perf_values)


class EnhancedAntennaForwardModel(nn.Module):
    """
    改进版天线正向预测模型
    考虑更多输入参数
    """

    def __init__(self, input_dim=11, output_dim=4):
        super(EnhancedAntennaForwardModel, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class EnhancedAntennaInverseModel(nn.Module):
    """
    改进版天线逆向设计模型
    输出更多详细参数
    """

    def __init__(self, input_dim=4, output_dim=11):
        super(EnhancedAntennaInverseModel, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, output_dim),
            nn.Tanh()  # 使用Tanh确保输出在[-1,1]范围内
        )

    def forward(self, x):
        return self.network(x)


class EnhancedAntennaDesignFramework:
    """
    改进版天线设计框架
    考虑介质板和接地部分的详细影响
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

    def prepare_data(self, num_samples=10000, test_ratio=0.2, batch_size=64):
        """准备训练和测试数据"""
        print("正在生成增强版天线数据集...")
        self.dataset = EnhancedAntennaDataset(num_samples=num_samples)

        # 划分训练集和测试集
        test_size = int(num_samples * test_ratio)
        train_size = num_samples - test_size

        train_dataset, test_dataset = data.random_split(
            self.dataset, [train_size, test_size]
        )

        self.train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"数据集准备完成: 训练集 {train_size} 样本, 测试集 {test_size} 样本")
        print(f"输入参数数量: {len(self.dataset.parameters)}")
        print(f"输出性能指标数量: 4 (谐振频率, 带宽, 增益, 回波损耗)")

    def build_models(self):
        """构建增强版正向和逆向模型"""
        print("正在构建增强版神经网络模型...")

        # 正向预测模型（输入11个参数，输出4个性能指标）
        self.forward_model = EnhancedAntennaForwardModel(input_dim=len(self.dataset.parameters), output_dim=4).to(
            self.device)

        # 逆向设计模型（输入4个性能指标，输出11个参数）
        self.inverse_model = EnhancedAntennaInverseModel(input_dim=4, output_dim=len(self.dataset.parameters)).to(
            self.device)

        print("模型构建完成")

    def train_forward_model(self, epochs=100, lr=0.001):
        """训练正向预测模型"""
        print("\n开始训练增强版正向预测模型...")

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.forward_model.parameters(), lr=lr)

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
                optimizer.step()

                total_loss += loss.item() * inputs.size(0)

            # 计算平均损失
            avg_loss = total_loss / len(self.train_loader.dataset)
            self.forward_history['loss'].append(avg_loss)

            # 验证
            val_loss = self.evaluate_model(self.forward_model, self.test_loader)
            self.forward_history['val_loss'].append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"正向模型 - 第 {epoch + 1}/{epochs} 轮: 训练损失 {avg_loss:.6f}, 验证损失 {val_loss:.6f}")

        print("正向预测模型训练完成")

    def train_inverse_model(self, epochs=100, lr=0.001):
        """训练逆向设计模型"""
        print("\n开始训练增强版逆向设计模型...")

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.inverse_model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.inverse_model.train()
            total_loss = 0

            for params, performances in self.train_loader:
                # 对于逆向模型，输入是性能指标，输出是结构参数
                performances, params = performances.to(self.device), params.to(self.device)

                # 前向传播
                outputs = self.inverse_model(performances)
                loss = criterion(outputs, params)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * performances.size(0)

            # 计算平均损失
            avg_loss = total_loss / len(self.train_loader.dataset)
            self.inverse_history['loss'].append(avg_loss)

            # 验证
            val_loss = self.evaluate_inverse_model(self.inverse_model, self.test_loader)
            self.inverse_history['val_loss'].append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"逆向模型 - 第 {epoch + 1}/{epochs} 轮: 训练损失 {avg_loss:.6f}, 验证损失 {val_loss:.6f}")

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
                # 输入是性能指标，目标是结构参数
                performances, params = performances.to(self.device), params.to(self.device)
                outputs = model(performances)
                loss = criterion(outputs, params)
                total_loss += loss.item() * performances.size(0)

        return total_loss / len(dataloader.dataset)

    def predict_performance(self, parameters):
        """
        预测天线性能
        parameters: 字典，包含详细的天线参数
        """
        self.forward_model.eval()

        # 归一化输入参数
        normalized_params = []
        param_keys = list(self.dataset.parameters.keys())
        for key in param_keys:
            if key in parameters:
                mean, std = self.dataset.param_stats[key]
                normalized_params.append((parameters[key] - mean) / std)
            else:
                # 如果缺少某些参数，使用默认值
                mean, _ = self.dataset.param_stats[key]
                normalized_params.append((mean - mean) / 1.0)  # 使用均值

        input_tensor = torch.tensor(normalized_params, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            normalized_output = self.forward_model(input_tensor)

        # 反归一化输出
        output_dict = {}
        perf_keys = ['resonance_frequency', 'bandwidth', 'gain', 's11']
        for i, key in enumerate(perf_keys):
            mean, std = self.dataset.perf_stats[key]
            output_dict[key] = normalized_output[0, i].item() * std + mean

        return output_dict

    def design_antenna(self, target_performances):
        """
        基于目标性能设计天线结构
        target_performances: 字典，包含期望的性能指标
        """
        self.inverse_model.eval()

        # 归一化目标性能指标
        normalized_targets = []
        perf_keys = ['resonance_frequency', 'bandwidth', 'gain', 's11']
        for key in perf_keys:
            if key in target_performances:
                mean, std = self.dataset.perf_stats[key]
                normalized_targets.append((target_performances[key] - mean) / std)
            else:
                # 如果缺少某些性能指标，使用默认值
                mean, _ = self.dataset.perf_stats[key]
                normalized_targets.append((mean - mean) / 1.0)

        input_tensor = torch.tensor(normalized_targets, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            normalized_output = self.inverse_model(input_tensor)

        # 反归一化输出参数
        output_dict = {}
        param_keys = list(self.dataset.parameters.keys())
        for i, key in enumerate(param_keys):
            mean, std = self.dataset.param_stats[key]
            output_dict[key] = normalized_output[0, i].item() * std + mean

        return output_dict

    def visualize_training_history(self):
        """可视化训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 正向模型训练历史
        ax1.plot(self.forward_history['loss'], label='训练损失')
        ax1.plot(self.forward_history['val_loss'], label='验证损失')
        ax1.set_title('增强版正向预测模型训练历史')
        ax1.set_xlabel('轮次')
        ax1.set_ylabel('损失')
        ax1.legend()
        ax1.grid(True)

        # 逆向模型训练历史
        ax2.plot(self.inverse_history['loss'], label='训练损失')
        ax2.plot(self.inverse_history['val_loss'], label='验证损失')
        ax2.set_title('增强版逆向设计模型训练历史')
        ax2.set_xlabel('轮次')
        ax2.set_ylabel('损失')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('./picture/enhanced_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("训练历史图已保存为 enhanced_training_history.png")

    def visualize_antenna_structure(self, parameters, save_path='./picture/enhanced_antenna_design.png'):
        """可视化详细的天线结构设计"""
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
        ax1.text(0, ground_width / 2 + 1, '接地平面', ha='center', va='center', color='gray')
        ax1.text(0, substrate_width / 2 + 1, '介质板', ha='center', va='center', color='green')
        ax1.text(0, patch_width / 2 + 1, '辐射贴片', ha='center', va='center', color='blue')

        # 侧视图
        substrate_thickness = parameters['substrate_thickness']
        ground_thickness = parameters['ground_thickness']

        # 接地平面侧视图
        ground_side = patches.Rectangle((-ground_length / 2, -ground_thickness),
                                        ground_length, ground_thickness,
                                        linewidth=1, edgecolor='gray', facecolor='lightgray', alpha=0.5)
        ax2.add_patch(ground_side)

        # 介质板侧视图
        substrate_side = patches.Rectangle((-substrate_length / 2, 0),
                                           substrate_length, substrate_thickness,
                                           linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.6)
        ax2.add_patch(substrate_side)

        # 贴片侧视图
        patch_side = patches.Rectangle((-patch_length / 2, substrate_thickness),
                                       patch_length, 0.1,  # 贴片厚度简化为0.1
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
        print(f"详细天线结构图已保存为 {save_path}")

    def save_models(self, save_dir='enhanced_antenna_models'):
        """保存训练好的模型"""
        os.makedirs(save_dir, exist_ok=True)

        # 保存正向模型
        forward_path = os.path.join(save_dir, 'enhanced_forward_model.pth')
        torch.save(self.forward_model.state_dict(), forward_path)

        # 保存逆向模型
        inverse_path = os.path.join(save_dir, 'enhanced_inverse_model.pth')
        torch.save(self.inverse_model.state_dict(), inverse_path)

        # 保存数据集统计信息
        stats_path = os.path.join(save_dir, 'enhanced_data_stats.npy')
        stats = {
            'param_stats': self.dataset.param_stats,
            'perf_stats': self.dataset.perf_stats,
            'param_names': list(self.dataset.parameters.keys()),
            'perf_names': ['resonance_frequency', 'bandwidth', 'gain', 's11']
        }
        np.save(stats_path, stats)

        print(f"增强版模型已保存到 {save_dir} 目录")

    def load_models(self, load_dir='enhanced_antenna_models'):
        """加载训练好的模型"""
        # 加载统计信息
        stats_path = os.path.join(load_dir, 'enhanced_data_stats.npy')
        stats = np.load(stats_path, allow_pickle=True).item()

        # 创建虚拟数据集用于反归一化
        self.dataset = EnhancedAntennaDataset(num_samples=1)
        self.dataset.param_stats = stats['param_stats']
        self.dataset.perf_stats = stats['perf_stats']
        self.dataset.parameters = {key: np.zeros(1) for key in stats['param_names']}
        self.dataset.performances = {key: np.zeros(1) for key in stats['perf_names']}

        # 加载正向模型
        forward_path = os.path.join(load_dir, 'enhanced_forward_model.pth')
        self.forward_model = EnhancedAntennaForwardModel(input_dim=len(stats['param_names']),
                                                         output_dim=len(stats['perf_names'])).to(self.device)
        self.forward_model.load_state_dict(torch.load(forward_path))

        # 加载逆向模型
        inverse_path = os.path.join(load_dir, 'enhanced_inverse_model.pth')
        self.inverse_model = EnhancedAntennaInverseModel(input_dim=len(stats['perf_names']),
                                                         output_dim=len(stats['param_names'])).to(self.device)
        self.inverse_model.load_state_dict(torch.load(inverse_path))

        print(f"增强版模型已从 {load_dir} 目录加载")


def main():
    """主函数"""
    print("=" * 60)
    print("增强版基于PyTorch的深度学习天线结构设计框架")
    print("版本: 2.0 - 考虑介质板和接地部分影响")
    print("=" * 60)

    # 创建增强版天线设计框架实例
    framework = EnhancedAntennaDesignFramework()

    # 步骤1: 准备数据（使用较少样本进行快速演示）
    framework.prepare_data(num_samples=5000, batch_size=32)

    # 步骤2: 构建模型
    framework.build_models()

    # 步骤3: 训练模型
    framework.train_forward_model(epochs=80, lr=0.001)
    framework.train_inverse_model(epochs=80, lr=0.001)

    # 步骤4: 可视化训练历史
    framework.visualize_training_history()

    # 步骤5: 保存模型
    framework.save_models()

    print("\n" + "=" * 60)
    print("增强版模型训练完成！开始演示天线设计功能...")
    print("=" * 60)

    # 演示1: 正向预测 - 已知结构参数预测性能
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

    # 演示2: 逆向设计 - 基于目标性能设计结构
    print("\n【演示2: 逆向设计】")
    target_performances = {
        'resonance_frequency': 2.4,  # 目标谐振频率 2.4GHz
        'bandwidth': 100,  # 目标带宽 100MHz
        'gain': 6.0,  # 目标增益 6dBi
        's11': -20.0  # 目标回波损耗 -20dB
    }

    designed_parameters = framework.design_antenna(target_performances)
    print(f"目标性能: {target_performances}")
    print(f"设计参数: {designed_parameters}")

    # 验证设计结果
    verified_performance = framework.predict_performance(designed_parameters)
    print(f"验证性能: {verified_performance}")

    # 可视化设计结果（合并参数和性能）
    design_with_performance = {**designed_parameters, **verified_performance}
    framework.visualize_antenna_structure(design_with_performance, './picture/enhanced_designed_antenna.png')

    print("\n" + "=" * 60)
    print("增强版天线设计演示完成！")
    print("生成的文件:")
    print("- enhanced_training_history.png: 训练历史图")
    print("- enhanced_designed_antenna.png: 详细天线设计结构图")
    print("- enhanced_antenna_models/: 增强版训练模型文件")
    print("=" * 60)


if __name__ == "__main__":
    main()