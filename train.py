from deep_learning_antenna_design import AntennaDesignFramework
from enhanced_antenna_design_framework import EnhancedAntennaDesignFramework

# 创建框架实例
framework = EnhancedAntennaDesignFramework()

# 准备数据
framework.prepare_data(num_samples=10000, batch_size=64)

# 构建模型
framework.build_models()

# 训练模型
framework.train_forward_model(epochs=100, lr=0.001)
framework.train_inverse_model(epochs=100, lr=0.001)

# 保存模型
framework.save_models()


# # 创建框架实例
# framework = AntennaDesignFramework()
#
# # 准备数据
# framework.prepare_data(num_samples=10000, batch_size=64)
#
# # 构建模型
# framework.build_models()
#
# # 训练模型
# framework.train_forward_model(epochs=100, lr=0.001)
# framework.train_inverse_model(epochs=100, lr=0.001)
#
# # 保存模型
# framework.save_models()