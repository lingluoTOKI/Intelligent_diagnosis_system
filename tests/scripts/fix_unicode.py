#!/usr/bin/env python3
"""
批量修复文件中的Unicode emoji问题
将所有emoji替换为兼容Windows GBK编码的文本
"""

import os
import re

# emoji替换映射
EMOJI_REPLACEMENTS = {
    '🏥': '[医院]',
    '🚀': '[启动]',
    '✅': '[成功]',
    '❌': '[错误]',
    '⚠️': '[警告]',
    '🔍': '[检查]',
    '📹': '[摄像头]',
    '🔬': '[诊断]',
    '🎮': '[命令]',
    '📊': '[统计]',
    '🌐': '[网络]',
    '📤': '[发送]',
    '📸': '[图像]',
    '📋': '[请求]',
    '💡': '[提示]',
    '🔄': '[处理]',
    '🛑': '[停止]',
    '📈': '[状态]',
    '⏱️': '[时间]',
    '🔗': '[连接]',
    '⏳': '[等待]',
    '🖥️': '[屏幕]',
    '📏': '[分辨率]',
    '📦': '[网络]',
    '🔸': '[停止]',
    '🖥': '[屏幕]',  # 单字节版本
    '🎯': '[目标]',
    '📝': '[文档]',
    '📱': '[开发板]',
    '🎤': '[录音]',
    '🗣️': '[说话]',
    '🔧': '[配置]',
    '🕒': '[时间]',
    '💾': '[保存]',
    '🛑': '[取消]',
    '🔬': '[诊断]',
    '🏠': '[本地]',
    '🟢': '[高]',
    '🟡': '[中]',
    '🔴': '[低]',
    '🚨': '[紧急]',
    '📭': '[空]',
    '📚': '[历史]',
    '📞': '[支持]',
    '💻': '[计算机]',
    '⚡': '[快速]',
    '🔥': '[优化]',
    '🎨': '[界面]',
    '🎵': '[音频]',
    '📺': '[显示]',
    '🔄': '[更新]',
    '⭐': '[重要]',
}

def fix_unicode_in_file(file_path):
    """修复单个文件中的Unicode问题"""
    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换emoji
        original_content = content
        for emoji, replacement in EMOJI_REPLACEMENTS.items():
            content = content.replace(emoji, replacement)
        
        # 如果有修改，保存文件
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ 修复了 {file_path}")
            return True
        else:
            print(f"- 跳过 {file_path} (无需修复)")
            return False
            
    except Exception as e:
        print(f"✗ 修复 {file_path} 失败: {e}")
        return False

def main():
    """主函数"""
    print("批量修复Unicode emoji问题")
    print("=" * 40)
    
    # 需要修复的文件列表
    files_to_fix = [
        '../pc_diagnosis_server.py',
        '../board_camera_integration.py',
        '../latency_optimizer.py',
        '../start_system_safe.py',
        '../link/wifi_pc_sender_with_mouse.py',
        '../link/wifi_pc_receiver_2.0.py'
    ]
    
    fixed_count = 0
    total_count = 0
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            total_count += 1
            if fix_unicode_in_file(file_path):
                fixed_count += 1
        else:
            print(f"! 文件不存在: {file_path}")
    
    print("=" * 40)
    print(f"修复完成: {fixed_count}/{total_count} 个文件")

if __name__ == "__main__":
    main()
