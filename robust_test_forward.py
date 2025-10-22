import os
from robust_antenna_design_framework import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

# 导入必要的类
from robust_antenna_design_framework import ImprovedAntennaDesignFramework, RobustAntennaDataset


class AntennaFrameworkTester:
    """天线设计框架测试器"""

    def __init__(self):
        self.framework = None
        self.test_results = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"测试设备: {self.device}")

    def load_trained_models(self, model_dir='robust_antenna_models'):
        """加载预训练模型"""
        print(f"\n正在加载预训练模型 from {model_dir}...")

        try:
            # 创建框架实例
            self.framework = ImprovedAntennaDesignFramework(device=self.device)

            # 加载模型
            self.framework.load_models(model_dir)

            print("✓ 模型加载成功")
            self.test_results['model_loading'] = 'success'
            return True

        except Exception as e:
            print(f"✗ 模型加载失败: {str(e)}")
            self.test_results['model_loading'] = 'failed'
            self.test_results['loading_error'] = str(e)
            return False

    def test_forward_prediction(self):
        """测试正向预测功能"""
        print("\n=== 测试正向预测功能 ===")

        try:
            # 测试案例1: 标准FR-4天线
            test_case1 = {
                'patch_length': 20.0,  # mm
                'patch_width': 15.0,  # mm
                'substrate_thickness': 1.6,  # mm (FR-4标准厚度)
                'substrate_epsr': 4.4,  # FR-4介电常数
                'substrate_length': 30.0,  # mm
                'substrate_width': 25.0,  # mm
                'ground_length': 30.0,  # mm
                'ground_width': 25.0,  # mm
                'ground_thickness': 0.035,  # mm (标准铜箔)
                'operating_frequency': 2.4,  # GHz (WiFi频率)
                'feed_position': 0.25  # 相对位置
            }

            print(f"测试案例1 - 输入参数: {test_case1}")
            performance1 = self.framework.predict_performance(test_case1)
            print(f"预测性能: {performance1}")

            # 测试案例2: 小型天线设计
            test_case2 = {
                'patch_length': 10.0,  # mm
                'patch_width': 8.0,  # mm
                'substrate_thickness': 0.8,  # mm
                'substrate_epsr': 3.38,  # Rogers 4350
                'substrate_length': 15.0,  # mm
                'substrate_width': 12.0,  # mm
                'ground_length': 15.0,  # mm
                'ground_width': 12.0,  # mm
                'ground_thickness': 0.017,  # mm
                'operating_frequency': 5.8,  # GHz (5G频率)
                'feed_position': 0.3  # 相对位置
            }

            print(f"\n测试案例2 - 输入参数: {test_case2}")
            performance2 = self.framework.predict_performance(test_case2)
            print(f"预测性能: {performance2}")

            # 保存测试结果
            self.test_results['./robust_test_result/forward_prediction'] = {
                'test_case1': {
                    'input': test_case1,
                    'output': performance1
                },
                'test_case2': {
                    'input': test_case2,
                    'output': performance2
                }
            }

            # 验证结果合理性
            self.validate_performance_results(performance1, "测试案例1")
            self.validate_performance_results(performance2, "测试案例2")

            print("✓ 正向预测测试通过")
            self.test_results['forward_prediction']['status'] = 'success'
            return True

        except Exception as e:
            print(f"✗ 正向预测测试失败: {str(e)}")
            self.test_results['forward_prediction'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False

    def test_inverse_design(self):
        """测试逆向设计功能"""
        print("\n=== 测试逆向设计功能 ===")

        try:
            # 目标性能1: WiFi天线
            target1 = {
                'resonance_frequency': 2.4,  # 目标谐振频率 2.4GHz
                'bandwidth': 80,  # 目标带宽 80MHz
                'gain': 5.0,  # 目标增益 5dBi
                's11': -18.0  # 目标回波损耗 -18dB
            }

            print(f"目标性能1: {target1}")
            design1 = self.framework.design_antenna(target1)
            print(f"设计参数: {design1}")

            # 验证设计结果
            verified_perf1 = self.framework.predict_performance(design1)
            print(f"验证性能: {verified_perf1}")

            # 目标性能2: 5G天线
            target2 = {
                'resonance_frequency': 3.5,  # 目标谐振频率 3.5GHz
                'bandwidth': 150,  # 目标带宽 150MHz
                'gain': 6.0,  # 目标增益 6dBi
                's11': -20.0  # 目标回波损耗 -20dB
            }

            print(f"\n目标性能2: {target2}")
            design2 = self.framework.design_antenna(target2)
            print(f"设计参数: {design2}")

            # 验证设计结果
            verified_perf2 = self.framework.predict_performance(design2)
            print(f"验证性能: {verified_perf2}")

            # 保存测试结果
            self.test_results['inverse_design'] = {
                'test_case1': {
                    'target': target1,
                    'design': design1,
                    'verified_performance': verified_perf1
                },
                'test_case2': {
                    'target': target2,
                    'design': design2,
                    'verified_performance': verified_perf2
                }
            }

            # 计算设计误差
            self.calculate_design_error(target1, verified_perf1, "测试案例1")
            self.calculate_design_error(target2, verified_perf2, "测试案例2")

            print("✓ 逆向设计测试通过")
            self.test_results['inverse_design']['status'] = 'success'
            return True

        except Exception as e:
            print(f"✗ 逆向设计测试失败: {str(e)}")
            self.test_results['inverse_design'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False

    def validate_performance_results(self, performance, case_name):
        """验证性能结果的合理性"""
        try:
            valid = True

            # 验证谐振频率在合理范围
            if not (0.1 <= performance['resonance_frequency'] <= 20):
                print(f"  ⚠ {case_name} - 谐振频率异常: {performance['resonance_frequency']} GHz")
                valid = False

            # 验证带宽在合理范围
            if not (5 <= performance['bandwidth'] <= 500):
                print(f"  ⚠ {case_name} - 带宽异常: {performance['bandwidth']} MHz")
                valid = False

            # 验证增益在合理范围
            if not (-10 <= performance['gain'] <= 20):
                print(f"  ⚠ {case_name} - 增益异常: {performance['gain']} dBi")
                valid = False

            # 验证回波损耗在合理范围
            if not (-40 <= performance['s11'] <= 0):
                print(f"  ⚠ {case_name} - 回波损耗异常: {performance['s11']} dB")
                valid = False

            if valid:
                print(f"  ✓ {case_name} - 性能结果合理")

            return valid

        except Exception as e:
            print(f"  ✗ {case_name} - 结果验证失败: {str(e)}")
            return False

    def calculate_design_error(self, target, actual, case_name):
        """计算设计误差"""
        try:
            errors = {}

            # 计算各指标误差
            errors['frequency_error'] = abs(target['resonance_frequency'] - actual['resonance_frequency'])
            errors['bandwidth_error'] = abs(target['bandwidth'] - actual['bandwidth'])
            errors['gain_error'] = abs(target['gain'] - actual['gain'])
            errors['s11_error'] = abs(target['s11'] - actual['s11'])

            # 计算相对误差
            errors['frequency_relative_error'] = errors['frequency_error'] / target['resonance_frequency'] * 100
            errors['bandwidth_relative_error'] = errors['bandwidth_error'] / target['bandwidth'] * 100

            print(f"  {case_name} - 设计误差:")
            print(f"    频率误差: {errors['frequency_error']:.2f} GHz ({errors['frequency_relative_error']:.1f}%)")
            print(f"    带宽误差: {errors['bandwidth_error']:.2f} MHz ({errors['bandwidth_relative_error']:.1f}%)")
            print(f"    增益误差: {errors['gain_error']:.2f} dBi")
            print(f"    回波损耗误差: {errors['s11_error']:.2f} dB")

            # 保存误差
            if 'errors' not in self.test_results:
                self.test_results['errors'] = {}
            self.test_results['errors'][case_name] = errors

            return errors

        except Exception as e:
            print(f"  ✗ 误差计算失败: {str(e)}")
            return None

    def test_model_performance(self):
        """测试模型整体性能"""
        print("\n=== 测试模型性能 ===")

        try:
            # 创建测试数据集
            test_dataset = RobustAntennaDataset(num_samples=1000, augment=False)
            print(f"创建测试数据集: {len(test_dataset)} 样本")

            # 转换为numpy数组
            all_params = []
            all_perfs = []

            for i in range(len(test_dataset)):
                params, perfs = test_dataset[i]
                all_params.append(params.numpy())
                all_perfs.append(perfs.numpy())

            all_params = np.array(all_params)
            all_perfs = np.array(all_perfs)

            # 转换为torch张量
            params_tensor = torch.tensor(all_params, dtype=torch.float32).to(self.device)
            perfs_tensor = torch.tensor(all_perfs, dtype=torch.float32).to(self.device)

            # 评估正向模型
            self.framework.forward_model.eval()
            with torch.no_grad():
                pred_perfs = self.framework.forward_model(params_tensor)

            # 计算MSE损失
            mse_loss = torch.mean((pred_perfs - perfs_tensor) ** 2).item()
            print(f"正向模型MSE损失: {mse_loss:.6f}")

            # 计算各指标的MAE
            mae_freq = torch.mean(torch.abs(pred_perfs[:, 0] - perfs_tensor[:, 0])).item()
            mae_bw = torch.mean(torch.abs(pred_perfs[:, 1] - perfs_tensor[:, 1])).item()
            mae_gain = torch.mean(torch.abs(pred_perfs[:, 2] - perfs_tensor[:, 2])).item()
            mae_s11 = torch.mean(torch.abs(pred_perfs[:, 3] - perfs_tensor[:, 3])).item()

            print(f"平均绝对误差:")
            print(f"  频率: {mae_freq:.4f}")
            print(f"  带宽: {mae_bw:.4f}")
            print(f"  增益: {mae_gain:.4f}")
            print(f"  回波损耗: {mae_s11:.4f}")

            # 保存性能指标
            self.test_results['model_performance'] = {
                'mse_loss': mse_loss,
                'mae_frequency': mae_freq,
                'mae_bandwidth': mae_bw,
                'mae_gain': mae_gain,
                'mae_s11': mae_s11,
                'test_samples': len(test_dataset)
            }

            print("✓ 模型性能测试通过")
            return True

        except Exception as e:
            print(f"✗ 模型性能测试失败: {str(e)}")
            self.test_results['model_performance'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False

    def generate_test_report(self):
        """生成测试报告"""
        print("\n=== 生成测试报告 ===")

        try:
            # 添加测试时间
            self.test_results['test_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.test_results['device'] = str(self.device)

            # 计算总体测试结果
            all_passed = all([
                self.test_results.get('model_loading') == 'success',
                self.test_results.get('forward_prediction', {}).get('status') == 'success',
                self.test_results.get('inverse_design', {}).get('status') == 'success',
                self.test_results.get('model_performance', {}).get('status') != 'failed'
            ])

            self.test_results['overall_result'] = 'PASS' if all_passed else 'FAIL'

            # 生成报告
            report = f"""
# 鲁棒版天线设计框架测试报告

## 测试摘要
- **测试时间**: {self.test_results['test_time']}
- **测试设备**: {self.test_results['device']}
- **总体结果**: {self.test_results['overall_result']}

## 详细测试结果

### 1. 模型加载
- **结果**: {self.test_results.get('model_loading', 'unknown')}
"""

            if self.test_results.get('model_loading') == 'failed':
                report += f"- **错误信息**: {self.test_results.get('loading_error', 'N/A')}\n"

            report += f"""
### 2. 正向预测功能
- **结果**: {self.test_results.get('forward_prediction', {}).get('status', 'unknown')}

"""

            if self.test_results.get('forward_prediction', {}).get('status') == 'success':
                case1 = self.test_results.get('forward_prediction', {}).get('test_case1', {})
                case2 = self.test_results.get('forward_prediction', {}).get('test_case2', {})

                report += f"""
#### 测试案例1 - 标准FR-4天线
- **输入参数**: {case1.get('input', 'N/A')}
- **预测性能**: {case1.get('output', 'N/A')}

#### 测试案例2 - 小型天线设计
- **输入参数**: {case2.get('input', 'N/A')}
- **预测性能**: {case2.get('output', 'N/A')}
"""
            else:
                report += f"- **错误信息**: {self.test_results.get('forward_prediction', {}).get('error', 'N/A')}\n"

            report += f"""
### 3. 逆向设计功能
- **结果**: {self.test_results.get('inverse_design', {}).get('status', 'unknown')}

"""

            if self.test_results.get('inverse_design', {}).get('status') == 'success':
                case1 = self.test_results.get('inverse_design', {}).get('test_case1', {})
                case2 = self.test_results.get('inverse_design', {}).get('test_case2', {})

                report += f"""
#### 测试案例1 - WiFi天线设计
- **目标性能**: {case1.get('target', 'N/A')}
- **设计参数**: {case1.get('design', 'N/A')}
- **验证性能**: {case1.get('verified_performance', 'N/A')}

#### 测试案例2 - 5G天线设计
- **目标性能**: {case2.get('target', 'N/A')}
- **设计参数**: {case2.get('design', 'N/A')}
- **验证性能**: {case2.get('verified_performance', 'N/A')}
"""
            else:
                report += f"- **错误信息**: {self.test_results.get('inverse_design', {}).get('error', 'N/A')}\n"

            report += f"""
### 4. 模型性能评估
"""

            performance = self.test_results.get('model_performance', {})
            if performance.get('status') != 'failed' and 'mse_loss' in performance:
                report += f"""
- **测试样本数**: {performance.get('test_samples', 'N/A')}
- **MSE损失**: {performance.get('mse_loss', 'N/A'):.6f}
- **平均绝对误差**:
  - 频率: {performance.get('mae_frequency', 'N/A'):.4f}
  - 带宽: {performance.get('mae_bandwidth', 'N/A'):.4f}
  - 增益: {performance.get('mae_gain', 'N/A'):.4f}
  - 回波损耗: {performance.get('mae_s11', 'N/A'):.4f}
"""
            else:
                report += f"- **结果**: {performance.get('status', 'unknown')}\n"
                if 'error' in performance:
                    report += f"- **错误信息**: {performance.get('error', 'N/A')}\n"

            # 保存报告
            with open('./robust_test_result/antenna_framework_test_report.md', 'w', encoding='utf-8') as f:
                f.write(report)

            # 保存详细结果为JSON
            with open('./robust_test_result/test_results.json', 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, indent=2)

            print("✓ 测试报告生成完成")
            print("  - antenna_framework_test_report.md")
            print("  - test_results.json")

            return True

        except Exception as e:
            print(f"✗ 报告生成失败: {str(e)}")
            return False

    def run_full_test_suite(self):
        """运行完整测试套件"""
        print("=" * 60)
        print("鲁棒版天线设计框架完整测试套件")
        print("=" * 60)

        # 运行所有测试
        tests = [
            ('模型加载', self.load_trained_models),
            ('正向预测', self.test_forward_prediction),
            ('逆向设计', self.test_inverse_design),
            ('模型性能', self.test_model_performance)
        ]

        all_passed = True

        for test_name, test_func in tests:
            print(f"\n测试: {test_name}")
            print("-" * 40)

            try:
                result = test_func()
                if not result:
                    all_passed = False
            except Exception as e:
                print(f"✗ {test_name} 异常失败: {str(e)}")
                all_passed = False

        # 生成报告
        self.generate_test_report()

        print("\n" + "=" * 60)
        print(f"测试总结: {'所有测试通过!' if all_passed else '部分测试失败!'}")
        print("=" * 60)

        return all_passed


if __name__ == "__main__":
    # 创建测试器并运行测试
    tester = AntennaFrameworkTester()
    success = tester.run_full_test_suite()

    # 根据测试结果设置退出码
    exit(0 if success else 1)