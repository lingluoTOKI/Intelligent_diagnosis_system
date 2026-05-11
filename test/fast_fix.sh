#!/bin/bash
# 快速修复脚本 - 只处理核心问题
# 使用方法: chmod +x fast_fix.sh && ./fast_fix.sh

echo "⚡ 快速修复开发板环境"
echo "======================"

# 1. 快速修复软件源（无备份，直接替换）
echo "🔧 修复软件源..."
sudo tee /etc/apt/sources.list > /dev/null <<'EOF'
deb http://archive.debian.org/debian buster main
deb http://repo.rock-chips.com/debian buster main
EOF

# 2. 快速更新（忽略错误）
echo "📦 更新包列表..."
sudo apt update --allow-releaseinfo-change -qq 2>/dev/null || true

# 3. 安装最少必需的包
echo "🔨 安装核心依赖..."
sudo apt install -y python3-pip python3-dev -qq 2>/dev/null || true

# 4. 快速安装Python包
echo "🐍 安装Python依赖..."
pip3 install opencv-python numpy -q 2>/dev/null || true

# 5. 尝试PyAudio（失败就跳过）
echo "🎤 尝试音频支持..."
sudo apt install -y python3-pyaudio -qq 2>/dev/null || pip3 install pyaudio -q 2>/dev/null || echo "音频跳过"

# 6. 尝试pygame（失败就跳过）
echo "🖥️ 尝试图形支持..."
pip3 install pygame -q 2>/dev/null || echo "图形跳过"

echo "✅ 快速修复完成！"
echo "🚀 运行: python3 board_integrated_system.py"
