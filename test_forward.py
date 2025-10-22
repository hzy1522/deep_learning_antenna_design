from enhanced_antenna_design_framework import EnhancedAntennaDesignFramework

# 创建框架实例
framework = EnhancedAntennaDesignFramework()

# 加载预训练模型
framework.load_models('enhanced_antenna_models')

# 基于目标性能设计天线

test_parameters = {
    'patch_length': 14.468,        # mm
    'patch_width': 23.677,         # mm
    'substrate_thickness': 3.911,  # mm (FR-4标准厚度)
    'substrate_epsr': 7.96,       # FR-4介电常数
    'substrate_length': 22.5767,    # mm
    'substrate_width': 38.6436,     # mm
    'ground_length': 22.1571,       # mm
    'ground_width': 39.0648,        # mm
    'ground_thickness': 0.035,   # mm (标准铜箔)
    'operating_frequency': 11.0728,  # GHz (WiFi频率)
    'feed_position': 0.6938        # 相对位置
}
antenna_params = framework.predict_performance(test_parameters)
print("预测性能参数:", antenna_params)
