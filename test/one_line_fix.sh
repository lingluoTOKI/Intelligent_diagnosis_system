#!/bin/bash
# 一行修复命令
echo 'deb http://archive.debian.org/debian buster main' | sudo tee /etc/apt/sources.list > /dev/null && sudo apt update -qq && sudo apt install -y python3-pip -qq && pip3 install opencv-python numpy -q && echo "✅ 基础修复完成，运行: python3 simple_start.py"
