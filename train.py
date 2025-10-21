from deep_learning_antenna_design import AntennaDesignFramework

# 创建框架实例
framework = AntennaDesignFramework()

# 加载预训练模型
framework.load_models('antenna_models')

# 基于目标性能设计天线
target_performances = {
    'resonance_frequency': 2.4,  # 2.4GHz
    'bandwidth': 80,             # 80MHz
    'gain': 5.0,                 # 5dBi
    's11': -15.0                 # -15dB
}

antenna_params = framework.design_antenna(target_performances)
print("设计参数:", antenna_params)