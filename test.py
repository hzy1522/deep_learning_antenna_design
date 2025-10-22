from deep_learning_antenna_design import AntennaDesignFramework
from enhanced_antenna_design_framework import EnhancedAntennaDesignFramework

# 创建框架实例
framework = EnhancedAntennaDesignFramework()

# 加载预训练模型
framework.load_models('enhanced_antenna_models')

# 基于目标性能设计天线
# target_performances = {
#     'resonance_frequency': 2.4,  # 2.4GHz
#     'bandwidth': 80,             # 80MHz
#     'gain': 5.0,                 # 5dBi
#     's11': -15.0                 # -15dB
# }
target_performances = {
    'resonance_frequency': 2.4,  # 目标谐振频率 2.4GHz
    'bandwidth': 100,            # 目标带宽 100MHz
    'gain': 6.0,                 # 目标增益 6dBi
    's11': -20.0                 # 目标回波损耗 -20dB
}

antenna_params = framework.design_antenna(target_performances)
print("设计参数:", antenna_params)

# # 创建框架实例
# framework = AntennaDesignFramework()
#
# # 加载预训练模型
# framework.load_models('antenna_models')
#
# # 基于目标性能设计天线
# # target_performances = {
# #     'resonance_frequency': 2.4,  # 2.4GHz
# #     'bandwidth': 80,             # 80MHz
# #     'gain': 5.0,                 # 5dBi
# #     's11': -15.0                 # -15dB
# # }
#
# target_performances = {
#     'resonance_frequency': 3.48,  # 2.4GHz
#     'bandwidth': 80,             # 80MHz
#     'gain': 5.0,                 # 5dBi
#     's11': -10.0                 # -15dB
# }
#
# antenna_params = framework.design_antenna(target_performances)
# print("设计参数:", antenna_params)