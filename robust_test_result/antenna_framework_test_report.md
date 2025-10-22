
# 鲁棒版天线设计框架测试报告

## 测试摘要
- **测试时间**: 2025-10-22 15:32:22
- **测试设备**: cuda
- **总体结果**: FAIL

## 详细测试结果

### 1. 模型加载
- **结果**: success

### 2. 正向预测功能
- **结果**: failed

- **错误信息**: 'forward_prediction'

### 3. 逆向设计功能
- **结果**: success


#### 测试案例1 - WiFi天线设计
- **目标性能**: {'resonance_frequency': 2.4, 'bandwidth': 80, 'gain': 5.0, 's11': -18.0}
- **设计参数**: {'patch_length': np.float64(37.09154938886127), 'patch_width': np.float64(20.771734007778534), 'substrate_thickness': np.float64(1.2402496143078916), 'substrate_epsr': np.float64(3.6702977250594477), 'substrate_length': np.float64(54.98215871122299), 'substrate_width': np.float64(30.74630381321419), 'ground_length': np.float64(55.13198065905178), 'ground_width': np.float64(30.846316227248792), 'ground_thickness': np.float64(0.02877544702830846), 'operating_frequency': np.float64(3.345891494593451), 'feed_position': np.float64(0.6555461233613809)}
- **验证性能**: {'resonance_frequency': np.float64(2.12443550546781), 'bandwidth': np.float64(398.78580483324924), 'gain': np.float64(-2.3555458733454597), 's11': np.float64(-3.0043607647343684)}

#### 测试案例2 - 5G天线设计
- **目标性能**: {'resonance_frequency': 3.5, 'bandwidth': 150, 'gain': 6.0, 's11': -20.0}
- **设计参数**: {'patch_length': np.float64(37.09154938886127), 'patch_width': np.float64(20.771734007778534), 'substrate_thickness': np.float64(1.2401270168294434), 'substrate_epsr': np.float64(3.6702977250594477), 'substrate_length': np.float64(54.982163673244244), 'substrate_width': np.float64(30.74630381321419), 'ground_length': np.float64(55.13197063231829), 'ground_width': np.float64(30.846316227248792), 'ground_thickness': np.float64(0.028903628645872412), 'operating_frequency': np.float64(3.343456697228612), 'feed_position': np.float64(0.6624040743748731)}
- **验证性能**: {'resonance_frequency': np.float64(2.1237843884703924), 'bandwidth': np.float64(398.8003231419744), 'gain': np.float64(-2.3553887829289675), 's11': np.float64(-3.0045545946387127)}

### 4. 模型性能评估

- **测试样本数**: 1000
- **MSE损失**: 0.497860
- **平均绝对误差**:
  - 频率: 0.0358
  - 带宽: 0.1239
  - 增益: 0.0485
  - 回波损耗: 0.7115
