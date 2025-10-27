"""
框架CST集成问题诊断工具
Framework CST Integration Diagnostic Tool

用于找出robust_antenna_design_framework_with_cst.py中的CST调用问题
"""

import os
import sys
import importlib.util
import traceback


def load_framework_module():
    """加载框架模块"""
    print("=" * 60)
    print("框架模块加载测试")
    print("=" * 60)

    framework_path = "robust_antenna_design_framework_with_cst.py"

    if not os.path.exists(framework_path):
        print(f"❌ 框架文件不存在: {framework_path}")
        return None

    print(f"✅ 找到框架文件: {framework_path}")

    try:
        # 尝试导入框架模块
        spec = importlib.util.spec_from_file_location("antenna_framework", framework_path)
        framework = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(framework)

        print("✅ 框架模块导入成功")
        return framework

    except Exception as e:
        print(f"❌ 框架导入失败: {str(e)}")
        print(f"详细错误: {traceback.format_exc()}")
        return None


def test_cst_simulator_directly():
    """直接测试CST仿真器类"""
    print("\n" + "=" * 60)
    print("直接测试CST仿真器类")
    print("=" * 60)

    # 直接创建CST仿真器（不通过完整框架）
    try:
        from robust_antenna_design_framework_with_cst import CSTAntennaSimulator

        print("✅ 成功导入CSTAntennaSimulator类")

        # 创建仿真器实例
        simulator = CSTAntennaSimulator(visible=True)
        print("✅ CSTAntennaSimulator实例创建成功")

        # 显示仿真器信息
        print(f"   COM可用: {simulator.com_available}")

        # 尝试连接
        if simulator.com_available:
            print("\n尝试连接CST...")
            if simulator.connect():
                print("✅ CST连接成功!")

                # 简单测试
                input("\n请确认CST已打开，按Enter继续...")

                # 创建测试项目
                if simulator.create_project("direct_test"):
                    print("✅ 项目创建成功")
                else:
                    print("❌ 项目创建失败")

                simulator.close()
            else:
                print("❌ CST连接失败")
        else:
            print("⚠️ COM接口不可用，跳过连接测试")

        return True

    except Exception as e:
        print(f"❌ CST仿真器测试失败: {str(e)}")
        print(f"详细错误: {traceback.format_exc()}")
        return False


def test_framework_cst_integration():
    """测试框架中的CST集成"""
    print("\n" + "=" * 60)
    print("测试框架中的CST集成")
    print("=" * 60)

    framework = load_framework_module()
    if not framework:
        return False

    try:
        # 创建框架实例
        print("\n创建框架实例（启用CST）...")
        antenna_framework = framework.ImprovedAntennaDesignFramework(use_cst=True)

        print("✅ 框架实例创建成功")
        print(f"   使用设备: {antenna_framework.device}")
        print(f"   CST集成: {'启用' if antenna_framework.use_cst else '禁用'}")

        # 检查CST仿真器
        if hasattr(antenna_framework, 'cst_simulator'):
            print(f"\nCST仿真器: {antenna_framework.cst_simulator}")
            print(f"COM可用: {antenna_framework.cst_simulator.com_available}")

            # 直接测试CST连接
            if antenna_framework.cst_simulator.com_available:
                print("\n直接测试CST连接...")
                if antenna_framework.cst_simulator.connect():
                    print("✅ CST连接成功!")
                    input("\n按Enter关闭CST...")
                    antenna_framework.cst_simulator.close()
                else:
                    print("❌ CST连接失败")
            else:
                print("⚠️ COM接口不可用")
        else:
            print("❌ 框架中没有CST仿真器属性")

        return True

    except Exception as e:
        print(f"❌ 框架集成测试失败: {str(e)}")
        print(f"详细错误: {traceback.format_exc()}")
        return False


def main():
    print("=" * 60)
    print("框架CST集成问题诊断工具")
    print("=" * 60)
    print("此工具用于诊断完整框架中的CST调用问题")
    print("=" * 60)

    # 检查操作系统
    if not sys.platform.startswith('win'):
        print(f"\n❌ 错误: 此工具仅支持Windows系统")
        print(f"   当前系统: {sys.platform}")
        return

    print(f"✅ 操作系统: Windows")

    # 运行测试
    tests = [
        ("直接测试CST仿真器类", test_cst_simulator_directly),
        ("测试框架中的CST集成", test_framework_cst_integration)
    ]

    print("\n" + "=" * 60)
    print("开始测试...")
    print("=" * 60)

    results = []
    for test_name, test_func in tests:
        print(f"\n【{test_name}】")
        success = test_func()
        results.append((test_name, success))
        print(f"{'✅' if success else '❌'} {test_name} {'通过' if success else '失败'}")

    # 生成报告
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    failed = sum(1 for _, success in results if not success)

    print(f"通过测试: {passed}/{len(results)}")
    print(f"失败测试: {failed}/{len(results)}")

    if failed > 0:
        print("\n❌ 发现问题:")
        for test_name, success in results:
            if not success:
                print(f"   - {test_name}")

        print("\n建议解决方案:")
        print("1. 检查框架代码中的CST调用部分")
        print("2. 确保所有CST操作都有错误处理")
        print("3. 验证CST仿真器类在框架中正确初始化")
        print("4. 检查权限和COM接口设置")
    else:
        print("\n✅ 所有测试通过!")
        print("   框架中的CST集成应该可以正常工作")


if __name__ == "__main__":
    main()