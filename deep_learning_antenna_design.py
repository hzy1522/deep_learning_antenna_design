import os
# 设置环境变量以避免OpenMP重复链接警告
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import interpolate
from datetime import datetime
import random

# 设置随机种子以确保可重复性
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

class AntennaDataset(data.Dataset):
    """
    天线数据集类，用于生成和加载天线设计数据

    输入参数:
    - length: 天线长度 (mm)
    - width: 天线宽度 (mm)
    - height: 天线高度 (mm)
    - dielectric_constant: 介质常数
    - frequency: 工作频率 (GHz)

    - length_sub: 介质板长度 (mm)
    - width_sub: 介质板宽度 (mm)
    - height_sub: 介质板厚度 (mm)

    -length_gnd:    接地板长度
    -width_gnd:     接地板宽度
    -height_gnd:    接地板厚度
    输出性能指标:
    - resonance_frequency: 谐振频率 (GHz)
    - bandwidth: 带宽 (MHz)
    - gain: 增益 (dBi)
    - s11: 回波损耗 (dB)
    """

    def __init__(self, num_samples=10000, normalize=True):
        self.num_samples = num_samples
        self.normalize = normalize

        # 生成天线参数数据
        self.generate_antenna_data()

        if normalize:
            self.normalize_data()

    def generate_antenna_data(self):
        """生成天线设计数据"""
        # 天线结构参数范围
        self.parameters = {
            'length': np.random.uniform(5, 50, self.num_samples),           # 长度 5-50mm
            'width': np.random.uniform(2, 20, self.num_samples),            # 宽度 2-20mm
            'height': np.random.uniform(0.5, 5, self.num_samples),          # 高度 0.5-5mm
            'dielectric_constant': np.random.uniform(2.2, 10.2, self.num_samples),  # 介质常数
            'frequency': np.random.uniform(1, 10, self.num_samples),        # 工作频率 1-10GHz
        }
        self.parameters = {
        # -------------------------------------------------------------------------------
            'length_sub': np.random.uniform(self.parameters['length'], 100, self.num_samples),  # 介质板长度(mm)
            'width_sub': np.random.uniform(self.parameters['width'], 100, self.num_samples),  # 介质板宽度(mm)
            'height_sub': np.random.uniform(self.parameters['height'], 5, self.num_samples),  # 介质板厚度(mm)

            'length_gnd': np.random.uniform(1, self.parameters['length_sub'], self.num_samples),  # 接地板长度(mm)
            'width_gnd': np.random.uniform(1, self.parameters['width_sub'], self.num_samples),    # 接地板宽度(mm)
            'height_gnd': np.random.uniform(0.01, 5, self.num_samples)                      # 接地板厚度(mm)
        }
        # 基于电磁理论计算性能指标（简化模型）
        self.performances = self.calculate_performances()

    def calculate_performances(self):
        """基于电磁理论计算天线性能指标"""
        params = self.parameters
        c = 3e8  # 光速

        # 计算谐振频率 (简化模型)
        wavelength = c / (params['frequency'] * 1e9)
        resonance_frequency = (c / (2 * params['length'] * 1e-3 * np.sqrt(params['dielectric_constant']))) / 1e9

        # 计算带宽 (简化模型)
        bandwidth = 50 + 2 * params['width'] + 0.5 * params['height'] + np.random.normal(0, 5, self.num_samples)
        bandwidth = np.clip(bandwidth, 20, 200)  # 限制在合理范围内

        # 计算增益 (简化模型)
        gain = 2.15 + 0.1 * params['length'] + 0.05 * params['width'] - 0.2 * params['height'] + np.random.normal(0, 0.3, self.num_samples)
        gain = np.clip(gain, 0, 10)  # 限制在合理范围内

        # 计算回波损耗 S11 (简化模型)
        mismatch = np.abs(resonance_frequency - params['frequency']) / params['frequency']
        s11 = -10 - 20 * np.exp(-5 * mismatch) + np.random.normal(0, 1, self.num_samples)
        s11 = np.clip(s11, -30, -5)  # 限制在合理范围内

        return {
            'resonance_frequency': resonance_frequency,
            'bandwidth': bandwidth,
            'gain': gain,
            's11': s11
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
        perf_values = np.array([self.performances[key][idx] for key in self.performances.keys()], dtype=np.float32)

        return torch.tensor(param_values), torch.tensor(perf_values)

class AntennaForwardModel(nn.Module):
    """
    天线正向预测模型
    输入: 天线结构参数 (5维)
    输出: 天线性能指标 (4维)
    """
    """
        天线正向预测模型
        输入: 天线结构参数 (5维)
        输出: 天线性能指标 (4维)
    """

    def __init__(self, input_dim=5, output_dim=4):
        super(AntennaForwardModel, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)

class AntennaInverseModel(nn.Module):
    """
    天线逆向设计模型
    输入: 期望的天线性能指标 (4维)
    输出: 对应的天线结构参数 (5维)
    """

    def __init__(self, input_dim=4, output_dim=5):
        super(AntennaInverseModel, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, output_dim),
            nn.Tanh()  # 使用Tanh确保输出在[-1,1]范围内，便于后续缩放
        )

    def forward(self, x):
        return self.network(x)

class AntennaDesignFramework:
    """
    天线设计框架类，包含完整的深度学习天线设计流程
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
        print("正在生成天线数据集...")
        self.dataset = AntennaDataset(num_samples=num_samples)

        # 划分训练集和测试集
        test_size = int(num_samples * test_ratio)
        train_size = num_samples - test_size

        train_dataset, test_dataset = data.random_split(
            self.dataset, [train_size, test_size]
        )

        self.train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"数据集准备完成: 训练集 {train_size} 样本, 测试集 {test_size} 样本")

    def build_models(self):
        """构建正向和逆向模型"""
        print("正在构建神经网络模型...")

        # 正向预测模型
        self.forward_model = AntennaForwardModel(input_dim=5, output_dim=4).to(self.device)

        # 逆向设计模型
        self.inverse_model = AntennaInverseModel(input_dim=4, output_dim=5).to(self.device)

        print("模型构建完成")

    def train_forward_model(self, epochs=100, lr=0.001):
        """训练正向预测模型"""
        print("\n开始训练正向预测模型...")

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
                print(f"正向模型 - 第 {epoch+1}/{epochs} 轮: 训练损失 {avg_loss:.6f}, 验证损失 {val_loss:.6f}")

        print("正向预测模型训练完成")

    def train_inverse_model(self, epochs=100, lr=0.001):
        """训练逆向设计模型"""
        print("\n开始训练逆向设计模型...")

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
                print(f"逆向模型 - 第 {epoch+1}/{epochs} 轮: 训练损失 {avg_loss:.6f}, 验证损失 {val_loss:.6f}")

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
        parameters: 字典，包含天线结构参数
        """
        self.forward_model.eval()

        # 归一化输入参数
        normalized_params = []
        for key in self.dataset.parameters.keys():
            mean, std = self.dataset.param_stats[key]
            normalized_params.append((parameters[key] - mean) / std)

        input_tensor = torch.tensor(normalized_params, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            normalized_output = self.forward_model(input_tensor)

        # 反归一化输出
        output_dict = {}
        for i, key in enumerate(self.dataset.performances.keys()):
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
        for key in self.dataset.performances.keys():
            mean, std = self.dataset.perf_stats[key]
            normalized_targets.append((target_performances[key] - mean) / std)

        input_tensor = torch.tensor(normalized_targets, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            normalized_output = self.inverse_model(input_tensor)

        # 反归一化输出参数
        output_dict = {}
        for i, key in enumerate(self.dataset.parameters.keys()):
            mean, std = self.dataset.param_stats[key]
            output_dict[key] = normalized_output[0, i].item() * std + mean

        return output_dict

    def visualize_training_history(self):
        """可视化训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 正向模型训练历史
        ax1.plot(self.forward_history['loss'], label='训练损失')
        ax1.plot(self.forward_history['val_loss'], label='验证损失')
        ax1.set_title('正向预测模型训练历史')
        ax1.set_xlabel('轮次')
        ax1.set_ylabel('损失')
        ax1.legend()
        ax1.grid(True)

        # 逆向模型训练历史
        ax2.plot(self.inverse_history['loss'], label='训练损失')
        ax2.plot(self.inverse_history['val_loss'], label='验证损失')
        ax2.set_title('逆向设计模型训练历史')
        ax2.set_xlabel('轮次')
        ax2.set_ylabel('损失')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("训练历史图已保存为 training_history.png")

    def visualize_antenna_structure(self, parameters, save_path='antenna_design.png'):
        """可视化天线结构设计"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # 绘制天线俯视图
        length = parameters['length']
        width = parameters['width']

        # 绘制天线主体
        antenna = patches.Rectangle((0, 0), length, width, linewidth=2,
                                   edgecolor='blue', facecolor='lightblue', alpha=0.7)
        ax.add_patch(antenna)

        # 绘制馈电点
        feed_point = patches.Circle((length/2, width/2), 0.5, color='red')
        ax.add_patch(feed_point)

        # 添加标注
        ax.text(length/2, width/2, '馈电点', ha='center', va='center', fontweight='bold')
        ax.text(length/2, -width/2, f'长度: {length:.1f}mm', ha='center', va='center')
        ax.text(-length/4, width/2, f'宽度: {width:.1f}mm', ha='center', va='center', rotation=90)

        ax.set_xlim(-length/2, length * 1.5)
        ax.set_ylim(-width, width * 1.5)
        ax.set_aspect('equal')
        ax.set_title(f'天线结构设计\n介质常数: {parameters["dielectric_constant"]:.2f}, 高度: {parameters["height"]:.2f}mm')
        ax.grid(True, alpha=0.3)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"天线结构图已保存为 {save_path}")

    def save_models(self, save_dir='antenna_models'):
        """保存训练好的模型"""
        os.makedirs(save_dir, exist_ok=True)

        # 保存正向模型
        forward_path = os.path.join(save_dir, 'forward_model.pth')
        torch.save(self.forward_model.state_dict(), forward_path)

        # 保存逆向模型
        inverse_path = os.path.join(save_dir, 'inverse_model.pth')
        torch.save(self.inverse_model.state_dict(), inverse_path)

        # 保存数据集统计信息
        stats_path = os.path.join(save_dir, 'data_stats.npy')
        stats = {
            'param_stats': self.dataset.param_stats,
            'perf_stats': self.dataset.perf_stats
        }
        np.save(stats_path, stats)

        print(f"模型已保存到 {save_dir} 目录")

    def load_models(self, load_dir='antenna_models'):
        """加载训练好的模型"""
        # 加载正向模型
        forward_path = os.path.join(load_dir, 'forward_model.pth')
        self.forward_model = AntennaForwardModel().to(self.device)
        self.forward_model.load_state_dict(torch.load(forward_path))

        # 加载逆向模型
        inverse_path = os.path.join(load_dir, 'inverse_model.pth')
        self.inverse_model = AntennaInverseModel().to(self.device)
        self.inverse_model.load_state_dict(torch.load(inverse_path))

        # 加载数据集统计信息
        stats_path = os.path.join(load_dir, 'data_stats.npy')
        stats = np.load(stats_path, allow_pickle=True).item()

        # 创建虚拟数据集用于反归一化
        self.dataset = AntennaDataset(num_samples=1)
        self.dataset.param_stats = stats['param_stats']
        self.dataset.perf_stats = stats['perf_stats']

        print(f"模型已从 {load_dir} 目录加载")

def main():
    """主函数"""
    print("="*60)
    print("基于PyTorch的深度学习天线结构设计框架")
    print("版本: 1.1 (修复OpenMP警告)")
    print("="*60)

    # 创建天线设计框架实例
    framework = AntennaDesignFramework()

    # 步骤1: 准备数据
    framework.prepare_data(num_samples=10000, batch_size=64)

    # 步骤2: 构建模型
    framework.build_models()

    # 步骤3: 训练模型
    framework.train_forward_model(epochs=100, lr=0.001)
    framework.train_inverse_model(epochs=100, lr=0.001)

    # 步骤4: 可视化训练历史
    framework.visualize_training_history()

    # 步骤5: 保存模型
    framework.save_models()

    print("\n" + "="*60)
    print("模型训练完成！开始演示天线设计功能...")
    print("="*60)

    # 演示1: 正向预测 - 已知结构参数预测性能
    print("\n【演示1: 正向预测】")
    test_parameters = {
        'length': 25.0,
        'width': 10.0,
        'height': 2.0,
        'dielectric_constant': 4.4,
        'frequency': 2.4
    }

    predicted_performance = framework.predict_performance(test_parameters)
    print(f"输入参数: {test_parameters}")
    print(f"预测性能: {predicted_performance}")

    # 演示2: 逆向设计 - 基于目标性能设计结构
    print("\n【演示2: 逆向设计】")
    target_performances = {
        'resonance_frequency': 2.4,  # 目标谐振频率 2.4GHz
        'bandwidth': 80,             # 目标带宽 80MHz
        'gain': 5.0,                 # 目标增益 5dBi
        's11': -15.0                 # 目标回波损耗 -15dB
    }

    designed_parameters = framework.design_antenna(target_performances)
    print(f"目标性能: {target_performances}")
    print(f"设计参数: {designed_parameters}")

    # 验证设计结果
    verified_performance = framework.predict_performance(designed_parameters)
    print(f"验证性能: {verified_performance}")

    # 可视化设计结果
    framework.visualize_antenna_structure(designed_parameters, 'designed_antenna.png')

    print("\n" + "="*60)
    print("天线设计演示完成！")
    print("生成的文件:")
    print("- training_history.png: 训练历史图")
    print("- designed_antenna.png: 天线设计结构图")
    print("- antenna_models/: 训练好的模型文件")
    print("="*60)

if __name__ == "__main__":
    main()