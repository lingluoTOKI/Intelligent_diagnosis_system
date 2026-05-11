#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医疗诊断系统智能启动器
自动处理依赖问题和环境配置
"""

import sys
import os
import subprocess

def print_banner():
    """显示启动横幅"""
    print("🏥" + "="*58 + "🏥")
    print("    开发板集成医疗诊断系统 - 智能启动器")
    print("🏥" + "="*58 + "🏥")
    print()

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 6):
        print("❌ Python版本过低，需要Python 3.6或更高版本")
        print(f"   当前版本: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✓ Python版本: {version.major}.{version.minor}.{version.micro}")
    return True

def check_and_install_opencv():
    """检查并尝试安装OpenCV"""
    try:
        import cv2
        print(f"✓ OpenCV 可用，版本: {cv2.__version__}")
        return True
    except ImportError:
        print("❌ OpenCV 未安装")
        
        # 尝试安装
        response = input("是否尝试自动安装OpenCV? (y/n): ")
        if response.lower() == 'y':
            try:
                print("📦 正在安装OpenCV...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
                
                # 重新检查
                import cv2
                print(f"✓ OpenCV 安装成功，版本: {cv2.__version__}")
                return True
            except Exception as e:
                print(f"❌ OpenCV 安装失败: {e}")
                return False
        return False

def check_optional_dependencies():
    """检查可选依赖"""
    dependencies = {
        'pyaudio': '语音功能',
        'pygame': '图形界面',
        'numpy': '数值计算',
        'requests': '网络通信'
    }
    
    available = {}
    missing = []
    
    for package, description in dependencies.items():
        try:
            __import__(package)
            print(f"✓ {package} 可用 - {description}")
            available[package] = True
        except ImportError:
            print(f"⚠️  {package} 不可用 - {description}将被禁用")
            available[package] = False
            missing.append(package)
    
    return available, missing

def setup_environment_variables(available_deps):
    """设置环境变量"""
    print("\n🔧 配置运行环境...")
    
    if not available_deps.get('pyaudio', False):
        os.environ['DISABLE_AUDIO'] = '1'
        print("   - 禁用音频功能")
    
    if not available_deps.get('pygame', False):
        os.environ['DISABLE_PYGAME'] = '1'
        print("   - 使用控制台界面")
    
    print("   - 环境变量配置完成")

def check_camera_device():
    """检查摄像头设备"""
    try:
        camera_devices = []
        for i in range(10):  # 检查 /dev/video0 到 /dev/video9
            device_path = f"/dev/video{i}"
            if os.path.exists(device_path):
                camera_devices.append(device_path)
        
        if camera_devices:
            print(f"✓ 摄像头设备: {', '.join(camera_devices)}")
            
            # 尝试打开摄像头测试
            try:
                import cv2
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    print("✓ 摄像头设备可以正常打开")
                    cap.release()
                    return True
                else:
                    print("⚠️  摄像头设备存在但无法打开")
                    return False
            except:
                print("⚠️  无法测试摄像头（OpenCV不可用）")
                return False
        else:
            print("⚠️  未检测到摄像头设备")
            return False
    
    except Exception as e:
        print(f"❌ 摄像头检查失败: {e}")
        return False

def install_missing_packages(missing_packages):
    """尝试安装缺失的包"""
    if not missing_packages:
        return
    
    print(f"\n📦 发现缺失的包: {', '.join(missing_packages)}")
    response = input("是否尝试自动安装这些包? (y/n): ")
    
    if response.lower() != 'y':
        print("跳过安装，系统将以降级模式运行")
        return
    
    # 包的安装命令映射
    install_commands = {
        'pyaudio': ['python3-pyaudio'],  # 优先使用系统包
        'pygame': ['pygame'],
        'numpy': ['numpy'],
        'requests': ['requests']
    }
    
    for package in missing_packages:
        try:
            if package in install_commands:
                print(f"📦 正在安装 {package}...")
                
                # 先尝试用apt安装系统包
                if package == 'pyaudio':
                    try:
                        subprocess.check_call(['sudo', 'apt', 'install', '-y', 'python3-pyaudio'], 
                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        print(f"✓ {package} 通过apt安装成功")
                        continue
                    except:
                        pass
                
                # 使用pip安装
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✓ {package} 安装成功")
        
        except Exception as e:
            print(f"❌ {package} 安装失败: {e}")

def check_system_requirements():
    """检查系统要求"""
    print("🔍 检查系统要求...")
    
    # 检查Python版本
    if not check_python_version():
        return False
    
    # 检查OpenCV（必需）
    if not check_and_install_opencv():
        print("❌ OpenCV是必需的依赖，无法继续")
        return False
    
    # 检查可选依赖
    available_deps, missing_packages = check_optional_dependencies()
    
    # 尝试安装缺失的包
    install_missing_packages(missing_packages)
    
    # 重新检查（安装后）
    available_deps, missing_packages = check_optional_dependencies()
    
    # 设置环境变量
    setup_environment_variables(available_deps)
    
    # 检查摄像头
    check_camera_device()
    
    return True

def run_medical_system():
    """运行医疗诊断系统"""
    try:
        print("\n🚀 启动医疗诊断系统...")
        print("-" * 50)
        
        # 检查主程序文件是否存在
        if not os.path.exists('../board_integrated_system.py'):
            print("❌ 未找到主程序文件 'board_integrated_system.py'")
            print("   请确保该文件在当前目录中")
            return False
        
        # 执行主程序
        exec(open('../board_integrated_system.py').read())
        
    except KeyboardInterrupt:
        print("\n\n👋 系统已被用户中断")
        return True
    except Exception as e:
        print(f"\n❌ 系统运行失败: {e}")
        print("\n🔧 故障排除建议:")
        print("   1. 检查所有依赖是否正确安装")
        print("   2. 确认摄像头设备可用")
        print("   3. 检查网络连接")
        print("   4. 查看详细错误日志")
        return False

def show_system_info():
    """显示系统信息"""
    print("\n📊 系统信息:")
    
    # 操作系统信息
    try:
        with open('/etc/os-release', 'r') as f:
            for line in f:
                if line.startswith('PRETTY_NAME='):
                    os_name = line.split('=', 1)[1].strip().strip('"')
                    print(f"   操作系统: {os_name}")
                    break
    except:
        print(f"   操作系统: {os.name}")
    
    # Python信息
    print(f"   Python版本: {sys.version.split()[0]}")
    
    # 架构信息
    import platform
    print(f"   系统架构: {platform.machine()}")

def main():
    """主函数"""
    print_banner()
    
    # 显示系统信息
    show_system_info()
    
    # 检查系统要求
    if not check_system_requirements():
        print("\n❌ 系统要求检查失败")
        print("   请参考 '开发板环境修复指南.md' 解决依赖问题")
        print("   或运行 './quick_fix.sh' 自动修复环境")
        return 1
    
    print("\n✅ 系统要求检查通过")
    
    # 运行系统
    if run_medical_system():
        print("\n✅ 系统正常退出")
        return 0
    else:
        print("\n❌ 系统异常退出")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n💥 启动器发生未处理的错误: {e}")
        print("   这可能是系统环境问题，请联系技术支持")
        sys.exit(1)
