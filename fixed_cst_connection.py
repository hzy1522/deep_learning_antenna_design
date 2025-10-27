"""
修复版CST连接代码
Fixed CST Connection Code

解决"Property 'CSTStudio.Application.Visible' can not be set"问题
"""

import os
import sys
import traceback
import time


class FixedCSTAntennaSimulator:
    """
    修复版CST天线仿真器
    解决Visible属性设置问题
    """

    def __init__(self, visible=True):
        self.cst = None
        self.visible = visible
        self.com_available = False
        self.connected = False

        self.check_com_availability()

    def check_com_availability(self):
        """检查COM接口可用性"""
        self.com_available = False

        # 检查操作系统
        if not sys.platform.startswith('win'):
            print("⚠️ 当前为非Windows系统，不支持CST COM接口")
            return

        # 检查pywin32
        try:
            import win32com.client
            self.com_available = True
            print("✅ Windows COM接口可用，支持CST调用")
        except ImportError:
            print("⚠️ pywin32未安装，无法调用CST")
            self.com_available = False

    def connect(self, max_attempts=3):
        """
        修复版CST连接方法
        移除Visible属性设置，增加重试机制
        """
        if not self.com_available:
            print("❌ COM接口不可用，无法连接CST")
            return False

        try:
            import win32com.client

            for attempt in range(1, max_attempts + 1):
                print(f"\n尝试连接CST (第{attempt}/{max_attempts}次)...")

                try:
                    # 尝试创建CST实例（不设置Visible属性）
                    start_time = time.time()

                    # 先尝试Dispatch
                    try:
                        self.cst = win32com.client.Dispatch("CSTStudio.Application")
                        print("✅ 使用Dispatch成功创建CST实例")
                    except:
                        # 如果失败，尝试DispatchEx
                        self.cst = win32com.client.DispatchEx("CSTStudio.Application")
                        print("✅ 使用DispatchEx成功创建CST实例")

                    connect_time = time.time() - start_time
                    print(f"⏱️ 创建CST实例耗时: {connect_time:.2f}秒")

                    # 验证CST对象
                    if self.cst is not None:
                        self.connected = True
                        print("✅ CST连接成功!")

                        # 尝试获取版本信息
                        try:
                            version = self.cst.Version
                            print(f"ℹ️ CST版本: {version}")
                        except Exception as e:
                            print(f"⚠️ 无法获取CST版本信息: {str(e)}")

                        # 即使无法设置Visible，连接也算成功
                        return True

                except Exception as e:
                    print(f"❌ 第{attempt}次连接失败: {str(e)}")
                    if attempt < max_attempts:
                        print("⏱️ 1秒后重试...")
                        time.sleep(1)

            # 所有尝试都失败
            print(f"❌ 所有{max_attempts}次连接尝试都失败")
            return False

        except Exception as e:
            print(f"❌ CST连接过程中发生错误: {str(e)}")
            print(f"📝 详细错误信息: {traceback.format_exc()}")
            return False

    def create_project(self, project_name="antenna_simulation"):
        """创建新的CST项目"""
        if not self.connected or not self.cst:
            print("❌ 未连接到CST，无法创建项目")
            return False

        try:
            print(f"\n创建CST项目: {project_name}")

            # 创建新项目
            project = self.cst.NewProject("MWS")  # Microwave Studio

            # 保存项目
            project_path = os.path.abspath(f"{project_name}.cst")
            project.SaveAs(project_path)

            print(f"✅ 项目已保存到: {project_path}")
            return project

        except Exception as e:
            print(f"❌ 创建项目失败: {str(e)}")
            print(f"📝 详细错误: {traceback.format_exc()}")
            return None

    def close(self):
        """关闭CST连接"""
        print("\n清理CST资源...")

        try:
            if hasattr(self, 'current_project') and self.current_project:
                try:
                    self.current_project.Close()
                    print("✅ 项目已关闭")
                except Exception as e:
                    print(f"⚠️ 关闭项目时出错: {str(e)}")

            if self.cst:
                try:
                    self.cst.Quit()
                    print("✅ CST已成功关闭")
                except Exception as e:
                    print(f"⚠️ 关闭CST时出错: {str(e)}")
                    print("ℹ️ 请手动关闭CST窗口")

            self.connected = False
            self.cst = None

        except Exception as e:
            print(f"❌ 清理资源时发生错误: {str(e)}")


def test_fixed_cst_connection():
    """测试修复版CST连接"""
    print("=" * 60)
    print("修复版CST连接测试工具")
    print("解决'Visible属性无法设置'问题")
    print("=" * 60)

    # 创建修复版CST仿真器
    simulator = FixedCSTAntennaSimulator()

    # 尝试连接
    if simulator.com_available:
        if simulator.connect():
            print("\n🎉 CST连接测试成功!")
            print("✅ 虽然无法设置Visible属性，但CST连接已成功")
            print("ℹ️ 您可以继续使用CST的其他功能")

            # 测试创建项目
            project = simulator.create_project("test_fixed_connection")
            if project:
                print("✅ 项目创建测试成功")

            # 关闭连接
            simulator.close()
        else:
            print("\n❌ CST连接测试失败")
    else:
        print("\n❌ COM接口不可用，无法进行CST连接测试")


def create_fixed_framework_patch():
    """创建框架修复补丁"""
    patch_code = '''
"""
CST连接问题修复补丁
Fixed CST Connection Patch

用于解决robust_antenna_design_framework_with_cst.py中的
"Property 'CSTStudio.Application.Visible' can not be set"问题
"""

import time
import traceback

class FixedCSTAntennaSimulator:
    """修复版CST天线仿真器"""

    def __init__(self, visible=True):
        self.cst = None
        self.visible = visible
        self.com_available = False
        self.connected = False

        self.check_com_availability()

    def check_com_availability(self):
        """检查COM接口可用性"""
        import sys
        self.com_available = False

        if not sys.platform.startswith('win'):
            return

        try:
            import win32com.client
            self.com_available = True
        except ImportError:
            self.com_available = False

    def connect(self, max_attempts=3):
        """修复版CST连接方法"""
        if not self.com_available:
            return False

        try:
            import win32com.client

            for attempt in range(1, max_attempts + 1):
                try:
                    # 尝试创建CST实例（不设置Visible属性）
                    try:
                        self.cst = win32com.client.Dispatch("CSTStudio.Application")
                    except:
                        self.cst = win32com.client.DispatchEx("CSTStudio.Application")

                    if self.cst is not None:
                        self.connected = True
                        return True

                except:
                    if attempt < max_attempts:
                        time.sleep(1)

            return False

        except:
            return False

    def close(self):
        """关闭CST连接"""
        try:
            if self.cst:
                self.cst.Quit()
        except:
            pass
        self.connected = False
        self.cst = None

# 使用方法：
# 将框架中的CSTAntennaSimulator替换为FixedCSTAntennaSimulator
# 在ImprovedAntennaDesignFramework的__init__方法中：
# self.cst_simulator = FixedCSTAntennaSimulator()
'''

    return patch_code


if __name__ == "__main__":
    # 运行修复版连接测试
    test_fixed_cst_connection()

    # 生成修复补丁
    patch = create_fixed_framework_patch()
    with open("cst_connection_fix.patch", "w", encoding='utf-8') as f:
        f.write(patch)

    print("\n📄 修复补丁已生成: cst_connection_fix.patch")
    print("ℹ️ 请按照补丁中的说明修改您的框架代码")