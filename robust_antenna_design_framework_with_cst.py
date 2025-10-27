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
import time

# 设置随机种子以确保可重复性
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


class CSTAntennaSimulator:
    """
    CST天线仿真器类
    提供与CST Microwave Studio的接口
    """

    def __init__(self, cst_path=None, visible=True):
        self.cst = None
        self.project = None
        self.cst_path = cst_path
        self.visible = visible
        self.com_available = False

        # 检查COM接口可用性（仅Windows）
        self.check_com_availability()

    def check_com_availability(self):
        """检查COM接口可用性"""
        import sys
        if sys.platform.startswith('win'):
            try:
                import win32com.client
                self.com_available = True
                print("✓ Windows COM接口可用，支持CST调用")
            except ImportError:
                print("✗ pywin32未安装，无法调用CST")
                self.com_available = False
        else:
            print("✗ 当前为非Windows系统，不支持CST COM接口")
            self.com_available = False

    def connect(self):
        """连接到CST"""
        if not self.com_available:
            return False

        try:
            import win32com.client
            self.cst = win32com.client.Dispatch("CSTStudio.Application")
            self.cst.Visible = self.visible
            print("✓ 成功连接到CST")
            return True
        except Exception as e:
            print(f"✗ 连接CST失败: {str(e)}")
            return False

    def create_project(self, project_name="antenna_simulation"):
        """创建新的CST项目"""
        if not self.com_available or not self.cst:
            return False

        try:
            self.project = self.cst.NewProject("MWS")  # Microwave Studio
            project_path = os.path.abspath(f"{project_name}.cst")
            self.project.SaveAs(project_path)
            print(f"✓ 创建CST项目: {project_path}")
            return True
        except Exception as e:
            print(f"✗ 创建项目失败: {str(e)}")
            return False

    def create_microstrip_antenna(self, params):
        """创建微带天线模型"""
        if not self.com_available or not self.project:
            return False

        try:
            modeler = self.project.Modeler
            modeler.Units = "mm"

            # 1. 创建接地平面
            ground = modeler.CreateRectangle(
                [0, 0, 0],
                [params['ground_length'], params['ground_width'], 0],
                "Ground"
            )
            ground.Material = "copper"

            # 2. 创建介质板
            substrate = modeler.CreateBox(
                [0, 0, 0],
                [params['substrate_length'], params['substrate_width'], params['substrate_thickness']],
                "Substrate"
            )

            # 设置介质板材料（尝试多种常见名称）
            substrate_materials = ["FR4_epoxy", "FR4", "FR-4", "Teflon"]
            material_set = False
            for mat_name in substrate_materials:
                try:
                    substrate.Material = mat_name
                    material_set = True
                    break
                except:
                    continue

            if not material_set:
                print("⚠ 未找到FR4材料，使用默认介质")
                substrate.Material = "dielectric"
                # 手动设置介电常数
                try:
                    mat_manager = self.project.Materials
                    mat = mat_manager.Item("dielectric")
                    mat.Epsilon = params['substrate_epsr']
                    mat.TangentDelta = 0.02  # FR4损耗角正切
                except:
                    pass

            # 3. 创建辐射贴片
            patch_x = (params['substrate_length'] - params['patch_length']) / 2
            patch_y = (params['substrate_width'] - params['patch_width']) / 2

            patch = modeler.CreateRectangle(
                [patch_x, patch_y, params['substrate_thickness']],
                [params['patch_length'], params['patch_width'], 0],
                "Patch"
            )
            patch.Material = "copper"

            # 4. 创建馈线
            feed_length = 10  # 馈线长度10mm
            feed_width = 2  # 馈线宽度2mm
            feed_x = patch_x + params['feed_position'] * params['patch_length']
            feed_y = (params['substrate_width'] - feed_width) / 2

            feed = modeler.CreateRectangle(
                [feed_x, feed_y, params['substrate_thickness']],
                [feed_length, feed_width, 0],
                "Feed"
            )
            feed.Material = "copper"

            # 5. 创建端口
            port_pos_x = feed_x + feed_length
            port = self.project.Ports.AddPort(
                [port_pos_x, params['substrate_width'] / 2, params['substrate_thickness'] / 2],
                [0, params['substrate_width'], 0],
                1
            )
            port.Name = "Port1"

            print("✓ 天线模型创建完成")
            return True

        except Exception as e:
            print(f"✗ 创建天线模型失败: {str(e)}")
            return False

    def set_simulation_parameters(self, start_freq=2.0, stop_freq=4.0, num_points=200):
        """设置仿真参数"""
        if not self.com_available or not self.project:
            return False

        try:
            solver = self.project.Solver
            solver.FrequencyRange = f"{start_freq}GHz"
            solver.FrequencyRange2 = f"{stop_freq}GHz"
            solver.SweepType = "Linear"
            solver.NumberOfFrequencyPoints = num_points
            solver.AdaptiveMesh = True  # 启用自适应网格
            print(f"✓ 仿真参数设置: {start_freq}-{stop_freq}GHz, {num_points}点")
            return True
        except Exception as e:
            print(f"✗ 设置仿真参数失败: {str(e)}")
            return False

    def run_simulation(self):
        """运行仿真"""
        if not self.com_available or not self.project:
            return False

        try:
            print("▶ 开始CST仿真...")
            start_time = time.time()

            solver = self.project.Solver
            solver.Run()

            end_time = time.time()
            print(f"✓ 仿真完成，耗时: {end_time - start_time:.1f}秒")
            return True
        except Exception as e:
            print(f"✗ 仿真失败: {str(e)}")
            return False

    def extract_results(self):
        """提取仿真结果"""
        if not self.com_available or not self.project:
            return None

        try:
            results = self.project.Results

            # 提取S参数
            s11_data = results.GetSParameterData("S1,1")
            frequencies = np.array(s11_data[0]) / 1e9  # 转换为GHz
            s11_mag = np.array(s11_data[1])

            # 找到谐振频率（S11最小点）
            min_s11_idx = np.argmin(s11_mag)
            resonance_freq = frequencies[min_s11_idx]

            # 计算-10dB带宽
            s11_threshold = -10
            bandwidth_points = frequencies[s11_mag <= s11_threshold]
            if len(bandwidth_points) > 0:
                bandwidth = bandwidth_points[-1] - bandwidth_points[0]
            else:
                bandwidth = 0.0  # 如果没有点低于-10dB，带宽设为0

            # 提取增益数据
            try:
                gain_data = results.GetGainData("GainTotal")
                gain_values = np.array(gain_data[1])
                peak_gain = np.max(gain_values)
            except:
                peak_gain = 0.0  # 如果无法提取增益，设为0

            results_dict = {
                'resonance_frequency': resonance_freq,
                'bandwidth': bandwidth * 1000,  # 转换为MHz
                'gain': peak_gain,
                's11': s11_mag[min_s11_idx],
                'frequency_points': frequencies,
                's11_curve': s11_mag
            }

            print(
                f"✓ 结果提取完成: f0={resonance_freq:.2f}GHz, BW={bandwidth * 1000:.0f}MHz, Gain={peak_gain:.1f}dBi, S11={s11_mag[min_s11_idx]:.1f}dB")
            return results_dict

        except Exception as e:
            print(f"✗ 提取结果失败: {str(e)}")
            return None

    def close(self):
        """关闭CST项目和连接"""
        if self.project:
            try:
                self.project.Save()
                self.project.Close()
                print("✓ 项目已保存并关闭")
            except:
                pass

        if self.cst:
            try:
                self.cst.Quit()
                print("✓ CST已关闭")
            except:
                pass

    def simulate_antenna(self, params, start_freq=2.0, stop_freq=4.0):
        """完整的天线仿真流程"""
        if not self.com_available:
            print("⚠ CST不可用，使用理论计算替代")
            return self.theoretical_calculation(params)

        try:
            # 连接CST
            if not self.connect():
                return self.theoretical_calculation(params)

            # 创建项目
            if not self.create_project():
                self.close()
                return self.theoretical_calculation(params)

            # 创建天线模型
            if not self.create_microstrip_antenna(params):
                self.close()
                return self.theoretical_calculation(params)

            # 设置仿真参数
            center_freq = params.get('operating_frequency', 2.4)
            if start_freq is None:
                start_freq = center_freq * 0.8
            if stop_freq is None:
                stop_freq = center_freq * 1.2

            if not self.set_simulation_parameters(start_freq, stop_freq):
                self.close()
                return self.theoretical_calculation(params)

            # 运行仿真
            if not self.run_simulation():
                self.close()
                return self.theoretical_calculation(params)

            # 提取结果
            results = self.extract_results()

            # 清理
            self.close()

            return results if results else self.theoretical_calculation(params)

        except Exception as e:
            print(f"⚠ 仿真过程出错: {str(e)}，使用理论计算替代")
            self.close()
            return self.theoretical_calculation(params)

    def theoretical_calculation(self, params):
        """理论计算作为备用方案"""
        c = 3e8  # 光速

        try:
            # 提取参数
            patch_length = params['patch_length']
            patch_width = params['patch_width']
            substrate_thickness = params['substrate_thickness']
            substrate_epsr = params['substrate_epsr']
            operating_freq = params.get('operating_frequency', 2.4)

            # 1. 计算谐振频率
            effective_epsr = (substrate_epsr + 1) / 2 + (substrate_epsr - 1) / 2 * \
                             np.power(1 + 12 * substrate_thickness / patch_width, -0.5)

            delta_l = substrate_thickness * (
                        0.412 * (effective_epsr + 0.3) * (patch_width / substrate_thickness + 0.264) /
                        ((effective_epsr - 0.258) * (patch_width / substrate_thickness + 0.8)))

            effective_length = patch_length + 2 * delta_l
            effective_length = max(effective_length, 1e-3)

            resonance_frequency = (c / (2 * effective_length * 1e-3 * np.sqrt(effective_epsr))) / 1e9
            resonance_frequency = max(resonance_frequency, 0.1)

            # 2. 计算带宽
            Q_radiation = (np.pi * np.sqrt(effective_epsr) * patch_width) / (2 * substrate_thickness)
            Q_dielectric = 1 / (
                np.tan(np.pi * substrate_epsr * substrate_thickness * operating_freq * 1e9 * 2 * np.pi / c))
            Q_conductor = (np.pi * np.sqrt(effective_epsr) * patch_width * np.sqrt(operating_freq * 1e9)) / (
                        2 * substrate_thickness * 6.62e4)

            Q_radiation = max(Q_radiation, 1)
            Q_dielectric = max(Q_dielectric, 1)
            Q_conductor = max(Q_conductor, 1)

            Q_total = 1 / (1 / Q_radiation + 1 / Q_dielectric + 1 / Q_conductor)
            fractional_bandwidth = 1.8 / Q_total
            bandwidth = fractional_bandwidth * resonance_frequency * 1000
            bandwidth = max(bandwidth, 10)

            # 3. 计算增益
            radiation_efficiency = Q_radiation / Q_total
            radiation_efficiency = max(min(radiation_efficiency, 0.99), 0.01)

            wavelength = c / (resonance_frequency * 1e9)
            directivity = (4 * np.pi * (patch_length * 1e-3) * (patch_width * 1e-3)) / (wavelength ** 2)
            directivity = max(directivity, 0.1)

            gain = 10 * np.log10(directivity * radiation_efficiency)
            gain = max(min(gain, 15), -5)

            # 4. 计算回波损耗
            Z0 = 377 / np.sqrt(effective_epsr)
            Z_patch = (120 * np.pi ** 2) / (120 * effective_epsr * (patch_width / patch_length) + 30 * np.pi)
            Z_patch = max(min(Z_patch, 200), 10)

            reflection_coefficient = (Z_patch - 50) / (Z_patch + 50)
            s11 = 20 * np.log10(np.abs(reflection_coefficient))

            freq_mismatch = abs(resonance_frequency - operating_freq) / resonance_frequency
            s11 += 10 * freq_mismatch * 15
            s11 = max(min(s11, -3), -35)

            return {
                'resonance_frequency': resonance_frequency,
                'bandwidth': bandwidth,
                'gain': gain,
                's11': s11,
                'method': 'theoretical'
            }

        except Exception as e:
            print(f"✗ 理论计算失败: {str(e)}")
            return {
                'resonance_frequency': 2.4,
                'bandwidth': 100,
                'gain': 2.0,
                's11': -15,
                'method': 'default'
            }


class RobustAntennaDataset(data.Dataset):
    """
    鲁棒版天线数据集类
    支持CST仿真和理论计算两种模式
    """

    def __init__(self, num_samples=15000, normalize=True, augment=True, use_cst=False):
        self.base_num_samples = num_samples
        self.normalize = normalize
        self.augment = augment
        self.use_cst = use_cst

        # 初始化CST仿真器
        self.cst_simulator = CSTAntennaSimulator(visible=True)
        # 新增调试输出（在这里添加）
        print(f"[调试] CST COM接口可用: {self.cst_simulator.com_available}")
        print(f"[调试] CST连接状态: {self.cst_simulator.cst is not None}")

        # 生成天线参数数据
        self.generate_antenna_data()

        if normalize:
            self.normalize_data()

    def generate_antenna_data(self):
        """生成天线参数数据"""
        print("正在生成天线数据集...")
        self.parameters = {
            # 贴片参数
            'patch_length': np.random.uniform(8, 45, self.base_num_samples),
            'patch_width': np.random.uniform(5, 25, self.base_num_samples),

            # 介质板参数
            'substrate_thickness': np.random.uniform(0.5, 4, self.base_num_samples),
            'substrate_epsr': np.random.uniform(2.5, 8.0, self.base_num_samples),
            'substrate_length': None,
            'substrate_width': None,

            # 接地平面参数
            'ground_length': None,
            'ground_width': None,
            'ground_thickness': np.random.choice([0.017, 0.035, 0.070], self.base_num_samples),

            # 工作参数
            'operating_frequency': np.random.uniform(1.0, 12, self.base_num_samples),
            'feed_position': np.random.uniform(0.2, 0.8, self.base_num_samples)
        }

        # 计算相关参数
        substrate_extension = np.random.uniform(1.1, 1.8, self.base_num_samples)
        self.parameters['substrate_length'] = self.parameters['patch_length'] * substrate_extension
        self.parameters['substrate_width'] = self.parameters['patch_width'] * substrate_extension

        ground_variation = np.random.uniform(0.9, 1.1, self.base_num_samples)
        self.parameters['ground_length'] = self.parameters['substrate_length'] * ground_variation
        self.parameters['ground_width'] = self.parameters['substrate_width'] * ground_variation

        # 计算性能指标（使用CST或理论计算）
        self.performances = self.calculate_performances()

        # 数据增强
        if self.augment and self.base_num_samples > 5000:
            self.augment_data()

    def calculate_performances(self):
        """计算天线性能指标"""
        num_samples = self.base_num_samples
        performances = {
            'resonance_frequency': np.zeros(num_samples),
            'bandwidth': np.zeros(num_samples),
            'gain': np.zeros(num_samples),
            's11': np.zeros(num_samples),
            'simulation_method': [''] * num_samples
        }

        print(f"正在计算性能指标 (共{num_samples}个样本)...")

        # 如果启用CST且可用，使用CST仿真
        if self.use_cst and self.cst_simulator.com_available:
            print("使用CST进行仿真计算（这将需要较长时间）...")

            # 为了效率，只对部分样本使用CST
            cst_samples = min(100, num_samples // 10)  # 最多100个CST样本
            theoretical_samples = num_samples - cst_samples

            print(f"计划使用CST仿真{cst_samples}个样本，理论计算{theoretical_samples}个样本")

            # CST仿真样本
            for i in range(cst_samples):
                if i % 10 == 0:
                    print(f"进度: {i}/{cst_samples} (CST仿真)")

                params = self._get_parameters_for_sample(i)
                result = self.cst_simulator.simulate_antenna(params)

                performances['resonance_frequency'][i] = result['resonance_frequency']
                performances['bandwidth'][i] = result['bandwidth']
                performances['gain'][i] = result['gain']
                performances['s11'][i] = result['s11']
                performances['simulation_method'][i] = result.get('method', 'unknown')

            # 理论计算样本
            for i in range(cst_samples, num_samples):
                if i % 100 == 0:
                    print(f"进度: {i}/{num_samples} (理论计算)")

                params = self._get_parameters_for_sample(i)
                result = self.cst_simulator.theoretical_calculation(params)

                performances['resonance_frequency'][i] = result['resonance_frequency']
                performances['bandwidth'][i] = result['bandwidth']
                performances['gain'][i] = result['gain']
                performances['s11'][i] = result['s11']
                performances['simulation_method'][i] = 'theoretical'

        else:
            # 全部使用理论计算
            for i in range(num_samples):
                if i % 100 == 0:
                    print(f"进度: {i}/{num_samples} (理论计算)")

                params = self._get_parameters_for_sample(i)
                result = self.cst_simulator.theoretical_calculation(params)

                performances['resonance_frequency'][i] = result['resonance_frequency']
                performances['bandwidth'][i] = result['bandwidth']
                performances['gain'][i] = result['gain']
                performances['s11'][i] = result['s11']
                performances['simulation_method'][i] = result.get('method', 'theoretical')

        print("性能指标计算完成")
        return performances

    def _get_parameters_for_sample(self, index):
        """获取单个样本的参数"""
        return {
            key: self.parameters[key][index] for key in self.parameters.keys()
        }

    def augment_data(self):
        """数据增强"""
        print("正在进行数据增强...")

        augment_ratio = 0.1
        augment_size = int(self.base_num_samples * augment_ratio)

        if augment_size == 0:
            return

        # 随机选择要增强的样本
        indices = np.random.choice(self.base_num_samples, augment_size, replace=False)

        # 创建增强参数
        augmented_params = {}
        for key in self.parameters:
            if key not in ['ground_thickness']:
                perturbation = np.random.uniform(0.95, 1.05, augment_size)
                augmented_params[key] = self.parameters[key][indices] * perturbation
            else:
                augmented_params[key] = self.parameters[key][indices]

        # 计算增强数据的性能指标
        augmented_performances = {
            'resonance_frequency': np.zeros(augment_size),
            'bandwidth': np.zeros(augment_size),
            'gain': np.zeros(augment_size),
            's11': np.zeros(augment_size),
            'simulation_method': ['augmented'] * augment_size
        }

        print(f"正在计算增强数据性能指标 ({augment_size}个样本)...")
        for i in range(augment_size):
            params = {key: augmented_params[key][i] for key in augmented_params.keys()}
            result = self.cst_simulator.theoretical_calculation(params)

            augmented_performances['resonance_frequency'][i] = result['resonance_frequency']
            augmented_performances['bandwidth'][i] = result['bandwidth']
            augmented_performances['gain'][i] = result['gain']
            augmented_performances['s11'][i] = result['s11']

        # 合并原始数据和增强数据
        for key in self.parameters:
            self.parameters[key] = np.concatenate([
                self.parameters[key],
                augmented_params[key]
            ])

        for key in self.performances:
            if key != 'simulation_method':
                self.performances[key] = np.concatenate([
                    self.performances[key],
                    augmented_performances[key]
                ])
            else:
                self.performances[key].extend(augmented_performances[key])

        self.base_num_samples += augment_size
        print(f"数据增强完成，总样本数：{self.base_num_samples}")

    def normalize_data(self):
        """数据归一化"""
        print("正在进行数据归一化...")
        self.param_stats = {}
        self.perf_stats = {}

        # 归一化参数
        for key in self.parameters:
            mean = np.mean(self.parameters[key])
            std = np.std(self.parameters[key])
            self.param_stats[key] = (mean, std)
            self.parameters[key] = (self.parameters[key] - mean) / std

        # 归一化性能指标
        for key in ['resonance_frequency', 'bandwidth', 'gain', 's11']:
            mean = np.mean(self.performances[key])
            std = np.std(self.performances[key])
            self.perf_stats[key] = (mean, std)
            self.performances[key] = (self.performances[key] - mean) / std

    def __len__(self):
        return self.base_num_samples

    def __getitem__(self, idx):
        """获取数据项"""
        param_values = np.array([self.parameters[key][idx] for key in self.parameters.keys()], dtype=np.float32)
        perf_values = np.array(
            [self.performances[key][idx] for key in ['resonance_frequency', 'bandwidth', 'gain', 's11']],
            dtype=np.float32)

        return torch.tensor(param_values), torch.tensor(perf_values)


class RobustAntennaForwardModel(nn.Module):
    """
    鲁棒版正向预测模型
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
    """早停机制"""

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
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class ImprovedAntennaDesignFramework:
    """
    改进版天线设计框架
    支持CST仿真集成
    """

    def __init__(self, device=None, use_cst=False):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_cst = use_cst
        print(f"使用设备: {self.device}")
        print(f"CST集成: {'启用' if use_cst else '禁用'}")

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

        # CST仿真器
        self.cst_simulator = CSTAntennaSimulator(visible=True)
        # 新增调试输出（在这里添加）
        print(f"[调试] CST COM接口可用: {self.cst_simulator.com_available}")
        print(f"[调试] CST连接状态: {self.cst_simulator.cst is not None}")

    def prepare_data(self, num_samples=15000, test_ratio=0.2, batch_size=64):
        """准备训练和测试数据"""
        print("正在生成鲁棒版天线数据集...")
        self.dataset = RobustAntennaDataset(
            num_samples=num_samples,
            augment=True,
            use_cst=self.use_cst
        )

        actual_dataset_size = len(self.dataset)
        test_size = int(actual_dataset_size * test_ratio)
        train_size = actual_dataset_size - test_size

        print(f"实际数据集大小: {actual_dataset_size}")
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
        """训练正向预测模型"""
        print("\n开始训练鲁棒版正向预测模型...")

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.forward_model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        # 早停机制
        early_stopping = EarlyStopping(patience=patience, verbose=True, path='./model_with_cst/best_forward_model.pt')

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
                nn.utils.clip_grad_norm_(self.forward_model.parameters(), max_norm=1.0)
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
        self.forward_model.load_state_dict(torch.load('./model_with_cst/best_forward_model.pt'))
        print("正向预测模型训练完成")

    def train_inverse_model(self, epochs=50, lr=0.001, patience=10):
        """训练逆向设计模型"""
        print("\n开始训练鲁棒版逆向设计模型...")

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.inverse_model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        # 早停机制
        early_stopping = EarlyStopping(patience=patience, verbose=True, path='./model_with_cst/best_inverse_model.pt')

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
                nn.utils.clip_grad_norm_(self.inverse_model.parameters(), max_norm=1.0)
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
        self.inverse_model.load_state_dict(torch.load('./model_with_cst/best_inverse_model.pt'))
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

    def simulate_design(self, parameters, use_cst=True):
        """使用CST仿真验证设计结果"""
        # 新增调试输出（在这里添加）
        print(f"[调试] 准备调用CST仿真，参数: {parameters}")
        print(f"[调试] 当前CST可用状态: {self.cst_simulator.com_available}")

        if use_cst and self.cst_simulator.com_available:
            try:  # 在这里添加try
                print("[调试] 开始CST仿真...")
                print("使用CST仿真验证设计...")
                result = self.cst_simulator.simulate_antenna(parameters)
                print("[调试] CST仿真成功")
                return result
            except Exception as e:
                print(f"[错误] CST仿真失败: {str(e)}")
                # 打印详细错误栈，便于调试
                import traceback
                traceback.print_exc()
        else:
            print("使用理论计算验证设计...")
            result = self.cst_simulator.theoretical_calculation(parameters)
            return result

    def visualize_training_history(self, save_path='./result_with_cst/robust_training_history.png'):
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

    def visualize_antenna_structure(self, parameters, save_path='robust_antenna_design.png'):
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

        # 初始化CST仿真器
        self.cst_simulator = CSTAntennaSimulator(visible=True)
        # 新增调试输出（在这里添加）
        print(f"[调试] CST COM接口可用: {self.cst_simulator.com_available}")
        print(f"[调试] CST连接状态: {self.cst_simulator.cst is not None}")

        print(f"鲁棒版模型已从 {load_dir} 目录加载")


def main():
    """主函数"""
    print("=" * 60)
    print("鲁棒版基于PyTorch的深度学习天线结构设计框架")
    print("版本: 3.0 - CST仿真集成版")
    print("=" * 60)

    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='鲁棒版天线设计框架')
    parser.add_argument('--use-cst', action='store_true', help='启用CST仿真（仅Windows）')
    parser.add_argument('--num-samples', type=int, default=15000, help='训练样本数量')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮次')

    args = parser.parse_args()

    # 创建改进版天线设计框架实例
    framework = ImprovedAntennaDesignFramework(use_cst=args.use_cst)

    # 步骤1: 准备数据
    framework.prepare_data(num_samples=args.num_samples, batch_size=64)

    # 步骤2: 构建模型
    framework.build_models(dropout_rate=0.25)

    # 步骤3: 训练模型
    framework.train_forward_model(epochs=args.epochs, lr=0.001, patience=8)
    framework.train_inverse_model(epochs=args.epochs, lr=0.001, patience=8)

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

    # 使用CST或理论计算验证
    verification_result = framework.simulate_design(test_parameters, use_cst=args.use_cst)
    print(f"验证性能: {verification_result}")

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
    verified_performance = framework.simulate_design(designed_parameters, use_cst=args.use_cst)
    print(f"验证性能: {verified_performance}")

    # 可视化设计结果
    design_with_performance = {**designed_parameters, **verified_performance}
    framework.visualize_antenna_structure(design_with_performance, './result_with_cst/robust_designed_antenna.png')

    print("\n" + "=" * 60)
    print("鲁棒版天线设计演示完成！")
    print("生成的文件:")
    print("- robust_training_history.png: 训练历史图")
    print("- robust_designed_antenna.png: 天线设计结构图")
    print("- robust_antenna_models/: 鲁棒版训练模型文件")
    print("=" * 60)

    # 输出改进总结
    print("\n【CST集成改进总结】")
    print("1. CST仿真集成:")
    print("   - 自动检测Windows COM接口可用性")
    print("   - 完整的CST建模和仿真流程")
    print("   - 支持微带天线的自动建模")
    print("   - 自动提取S参数、增益等关键指标")
    print("\n2. 智能降级机制:")
    print("   - Windows系统: 优先使用CST仿真")
    print("   - 非Windows系统: 自动使用理论计算")
    print("   - CST连接失败时: 自动切换到理论计算")
    print("\n3. 数据生成优化:")
    print("   - 支持CST仿真数据和理论数据混合训练")
    print("   - 提高模型的工程实用性和准确性")
    print("   - 为实际应用提供更可靠的预测结果")


if __name__ == "__main__":
    main()