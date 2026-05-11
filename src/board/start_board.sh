#!/bin/bash
# 开发板一键启动脚本

echo "[启动] 开发板医疗诊断系统..."

# 检查网络连接
ping -c 1 198.18.0.1 > /dev/null
if [ $? -ne 0 ]; then
    echo "[错误] 无法连接到PC端: 198.18.0.1"
    exit 1
fi

echo "[成功] 网络连接正常"

# 启动屏幕接收 (后台)
if [ -f "link/wifi_pc_receiver_2.0.py" ]; then
    cd link
    python wifi_pc_receiver_2.0.py &
    echo "[成功] 屏幕接收服务已启动"
    cd ..
fi

# 等待屏幕服务启动
sleep 3

# 启动摄像头诊断服务
if [ -f "board_camera_integration.py" ]; then
    python board_camera_integration.py
else
    echo "[错误] 找不到摄像头诊断程序"
fi
