"""
简化版CST连接测试脚本
Simplified CST Connection Test Script

适用于Windows环境下的CST集成测试
"""

import os
import sys
import traceback

def main():
    print("="*60)
    print("简化版CST连接测试工具")
    print("="*60)

    # 检查操作系统
    if not sys.platform.startswith('win'):
        print("❌ 错误: 此脚本仅支持Windows系统")
        print(f"   当前系统: {sys.platform}")
        print("   请在Windows环境下运行CST集成功能")
        return

    print(f"✅ 操作系统: Windows ({sys.platform})")

    # 检查pywin32
    try:
        import win32com.client
        print("✅ pywin32库已安装")
    except ImportError:
        print("❌ pywin32库未安装")
        print("   请安装: pip install pywin32")
        return

    # 尝试连接CST
    print("\n尝试连接CST...")
    try:
        # 尝试创建CST实例
        cst = win32com.client.Dispatch("CSTStudio.Application")
        print("✅ 成功连接到CST!")

        # 获取CST版本信息
        try:
            version = cst.Version
            print(f"   CST版本: {version}")
        except:
            print("   无法获取CST版本信息")

        # 显示CST界面
        try:
            cst.Visible = True
            print("✅ CST界面已显示")
        except:
            print("⚠️  无法设置CST界面可见性")

        # 等待用户确认
        input("\n请确认CST窗口已打开，然后按Enter键继续...")

        # 关闭CST
        try:
            cst.Quit()
            print("✅ CST已成功关闭")
        except:
            print("⚠️  关闭CST时出现问题")
            print("   请手动关闭CST窗口")

        print("\n" + "="*60)
        print("🎉 CST连接测试成功完成!")
        print("🎉 您的环境支持CST集成功能")
        print("="*60)

    except Exception as e:
        print(f"❌ CST连接失败: {str(e)}")
        print("\n详细错误信息:")
        traceback.print_exc()

        print("\n" + "="*60)
        print("问题分析和解决方案:")
        print("="*60)

        # 常见错误分析
        error_str = str(e).lower()
        if "class not registered" in error_str:
            print("🔍 错误类型: Class not registered")
            print("   可能原因:")
            print("   1. CST未正确安装")
            print("   2. CST COM组件未注册")
            print("   3. CST版本不支持自动化")
            print("   解决方案: 重新安装CST并确保选择COM组件")

        elif "access is denied" in error_str:
            print("🔍 错误类型: Access is denied")
            print("   可能原因: 权限不足")
            print("   解决方案: 以管理员身份运行Python")

        elif "operation unavailable" in error_str:
            print("🔍 错误类型: Operation unavailable")
            print("   可能原因: CST正在运行或被占用")
            print("   解决方案: 关闭所有CST实例后重试")

        else:
            print("🔍 未知错误")
            print("   请检查:")
            print("   1. CST是否已正确安装")
            print("   2. 是否有管理员权限")
            print("   3. CST版本是否支持自动化")

        print("\n建议:")
        print("1. 确保CST Microwave Studio已安装（非学生版）")
        print("2. 以管理员身份运行命令提示符")
        print("3. 在管理员模式下运行Python脚本")
        print("4. 检查CST安装时是否选择了'COM Interface'组件")

if __name__ == "__main__":
    main()