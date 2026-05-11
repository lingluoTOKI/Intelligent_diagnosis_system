#!/bin/bash
# 开发板环境快速修复脚本
# 使用方法: chmod +x quick_fix.sh && ./quick_fix.sh

set -e  # 遇到错误时停止

echo "🏥 开发板医疗诊断系统环境修复"
echo "=================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 1. 备份原始软件源
log_info "备份原始软件源..."
if [ -f /etc/apt/sources.list ]; then
    sudo cp /etc/apt/sources.list /etc/apt/sources.list.backup.$(date +%Y%m%d_%H%M%S)
    log_info "原始源已备份"
else
    log_warn "未找到原始源文件"
fi

# 2. 写入新的软件源配置
log_info "更新软件源配置..."
sudo tee /etc/apt/sources.list > /dev/null <<'EOF'
# Debian Buster 官方档案源
deb http://archive.debian.org/debian buster main contrib non-free
deb http://archive.debian.org/debian-security buster/updates main contrib non-free

# Rock-chips源
deb http://repo.rock-chips.com/debian buster main
EOF

# 3. 更新包列表
log_info "更新软件包列表..."
if sudo apt update --allow-releaseinfo-change; then
    log_info "软件包列表更新成功"
else
    log_warn "软件包列表更新部分失败，尝试替代源..."
    
    # 尝试使用中科大镜像源
    sudo tee /etc/apt/sources.list > /dev/null <<'EOF'
# 中科大镜像源
deb http://mirrors.ustc.edu.cn/debian buster main contrib non-free
deb http://mirrors.ustc.edu.cn/debian buster-updates main contrib non-free
deb http://mirrors.ustc.edu.cn/debian-security buster/updates main contrib non-free

# Rock-chips源
deb http://repo.rock-chips.com/debian buster main
EOF
    
    if sudo apt update; then
        log_info "使用中科大镜像源更新成功"
    else
        log_error "软件源更新失败，请检查网络连接"
        exit 1
    fi
fi

# 4. 安装基础开发工具
log_info "安装基础开发工具..."
if sudo apt install -y build-essential python3-dev python3-pip; then
    log_info "基础开发工具安装成功"
else
    log_error "基础开发工具安装失败"
    exit 1
fi

# 5. 尝试安装音频相关依赖
log_info "尝试安装音频依赖..."
AUDIO_DEPS="portaudio19-dev libasound2-dev libpulse-dev"
AUDIO_SUCCESS=false

for dep in $AUDIO_DEPS; do
    if sudo apt install -y $dep 2>/dev/null; then
        log_info "成功安装 $dep"
        AUDIO_SUCCESS=true
        break
    else
        log_warn "无法安装 $dep，尝试下一个..."
    fi
done

if [ "$AUDIO_SUCCESS" = false ]; then
    log_warn "所有音频依赖安装失败，将在无音频模式下运行"
fi

# 6. 安装Python基础依赖
log_info "安装Python基础依赖..."
pip3 install --upgrade pip

PYTHON_DEPS="opencv-python numpy requests datetime"
for dep in $PYTHON_DEPS; do
    if pip3 install $dep; then
        log_info "成功安装 $dep"
    else
        log_warn "安装 $dep 失败"
    fi
done

# 7. 尝试安装PyAudio
log_info "尝试安装PyAudio..."
PYAUDIO_SUCCESS=false

# 方法1: 使用apt安装预编译版本
if sudo apt install -y python3-pyaudio 2>/dev/null; then
    log_info "通过apt成功安装PyAudio"
    PYAUDIO_SUCCESS=true
else
    # 方法2: 使用pip编译安装
    if pip3 install pyaudio 2>/dev/null; then
        log_info "通过pip成功安装PyAudio"
        PYAUDIO_SUCCESS=true
    else
        log_warn "PyAudio安装失败，语音功能将不可用"
    fi
fi

# 8. 尝试安装pygame
log_info "尝试安装pygame..."
PYGAME_SUCCESS=false

if pip3 install pygame 2>/dev/null; then
    log_info "pygame安装成功"
    PYGAME_SUCCESS=true
else
    # 尝试通过apt安装
    if sudo apt install -y python3-pygame 2>/dev/null; then
        log_info "通过apt成功安装pygame"
        PYGAME_SUCCESS=true
    else
        log_warn "pygame安装失败，将使用控制台模式"
    fi
fi

# 9. 测试系统组件
echo ""
log_info "测试系统组件..."

# 测试OpenCV
if python3 -c "import cv2; print('OpenCV版本:', cv2.__version__)" 2>/dev/null; then
    log_info "✓ OpenCV 可用"
else
    log_error "✗ OpenCV 不可用"
fi

# 测试摄像头
if python3 -c "import cv2; cap = cv2.VideoCapture(0); print('摄像头可用:', cap.isOpened()); cap.release()" 2>/dev/null | grep -q "True"; then
    log_info "✓ 摄像头设备可用"
else
    log_warn "✗ 摄像头设备不可用"
fi

# 测试PyAudio
if python3 -c "import pyaudio; print('PyAudio可用')" 2>/dev/null; then
    log_info "✓ PyAudio 可用"
else
    log_warn "✗ PyAudio 不可用（语音功能将被禁用）"
fi

# 测试pygame
if python3 -c "import pygame; print('pygame可用')" 2>/dev/null; then
    log_info "✓ pygame 可用"
else
    log_warn "✗ pygame 不可用（将使用控制台界面）"
fi

# 10. 创建启动脚本
log_info "创建启动脚本..."
cat > run_medical_system.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医疗诊断系统启动脚本
自动检测环境并调整功能
"""

import sys
import os

def check_dependencies():
    """检查依赖并设置环境变量"""
    print("🔍 检查系统依赖...")
    
    # 检查PyAudio
    try:
        import pyaudio
        print("✓ PyAudio 可用")
    except ImportError:
        print("✗ PyAudio 不可用，禁用语音功能")
        os.environ['DISABLE_AUDIO'] = '1'
    
    # 检查pygame
    try:
        import pygame
        print("✓ pygame 可用")
    except ImportError:
        print("✗ pygame 不可用，使用控制台模式")
        os.environ['DISABLE_PYGAME'] = '1'
    
    # 检查OpenCV
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__} 可用")
    except ImportError:
        print("✗ OpenCV 不可用！")
        sys.exit(1)
    
    print("🚀 启动医疗诊断系统...\n")

if __name__ == "__main__":
    check_dependencies()
    
    # 导入并运行主系统
    try:
        exec(open('board_integrated_system.py').read())
    except FileNotFoundError:
        print("❌ 未找到 board_integrated_system.py 文件")
        print("请确保该文件在当前目录中")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 系统启动失败: {e}")
        sys.exit(1)
EOF

chmod +x run_medical_system.py
log_info "启动脚本已创建: run_medical_system.py"

# 11. 创建环境配置文件
log_info "创建环境配置文件..."
cat > system_config.txt << EOF
# 开发板医疗诊断系统配置
# 生成时间: $(date)

[系统信息]
操作系统: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)
Python版本: $(python3 --version)
架构: $(uname -m)

[已安装组件]
OpenCV: $(python3 -c "import cv2; print(cv2.__version__)" 2>/dev/null || echo "未安装")
PyAudio: $(python3 -c "import pyaudio; print('已安装')" 2>/dev/null || echo "未安装")
pygame: $(python3 -c "import pygame; print('已安装')" 2>/dev/null || echo "未安装")

[设备状态]
摄像头设备: $(ls /dev/video* 2>/dev/null || echo "未检测到")

[建议配置]
$(if [ "$PYAUDIO_SUCCESS" = false ]; then echo "- 语音功能已禁用"; fi)
$(if [ "$PYGAME_SUCCESS" = false ]; then echo "- 图形界面已禁用，使用控制台模式"; fi)
EOF

# 12. 输出总结
echo ""
echo "=================================="
log_info "环境修复完成！"
echo ""
echo "📝 总结:"
echo "  - 软件源已更新"
echo "  - 基础开发工具已安装"
if [ "$AUDIO_SUCCESS" = true ]; then
    echo "  - 音频依赖已安装"
else
    echo "  - 音频依赖安装失败（语音功能将被禁用）"
fi
if [ "$PYAUDIO_SUCCESS" = true ]; then
    echo "  - PyAudio 可用"
else
    echo "  - PyAudio 不可用"
fi
if [ "$PYGAME_SUCCESS" = true ]; then
    echo "  - pygame 可用"
else
    echo "  - pygame 不可用（控制台模式）"
fi
echo ""
echo "🚀 启动系统:"
echo "   python3 run_medical_system.py"
echo ""
echo "📋 查看详细配置:"
echo "   cat system_config.txt"
echo ""
log_info "如有问题，请查看 开发板环境修复指南.md"
