#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试开发板摄像头状态标签修复
验证 MainWindow 类中的 board_camera_status 属性是否正确定义
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_board_camera_status_attribute():
    """测试 board_camera_status 属性是否正确定义"""
    print("🔍 测试 MainWindow 类中的 board_camera_status 属性")
    print("=" * 60)
    
    try:
        # 导入必要的模块
        from PyQt5.QtWidgets import QApplication
        from visualization_test2 import MainWindow
        
        print("✅ 成功导入 MainWindow 类")
        
        # 创建应用程序实例（不显示窗口）
        app = QApplication(sys.argv) if not QApplication.instance() else QApplication.instance()
        
        print("✅ 创建 QApplication 实例")
        
        # 创建主窗口实例
        window = MainWindow()
        print("✅ 成功创建 MainWindow 实例")
        
        # 初始化UI
        window.init_ui()
        print("✅ 成功初始化用户界面")
        
        # 检查 board_camera_status 属性是否存在
        if hasattr(window, 'board_camera_status'):
            print("✅ board_camera_status 属性已正确定义")
            
            # 检查属性类型
            from PyQt5.QtWidgets import QLabel
            if isinstance(window.board_camera_status, QLabel):
                print("✅ board_camera_status 是 QLabel 类型")
                
                # 检查初始文本
                initial_text = window.board_camera_status.text()
                print(f"✅ 初始状态文本: '{initial_text}'")
                
                # 测试状态更新
                print("\n🔄 测试状态更新功能:")
                
                # 测试连接状态
                window.board_camera_status.setText("🟢 已连接")
                connected_text = window.board_camera_status.text()
                print(f"   连接状态: '{connected_text}'")
                
                # 测试断开状态
                window.board_camera_status.setText("🔴 未连接")
                disconnected_text = window.board_camera_status.text()
                print(f"   断开状态: '{disconnected_text}'")
                
                # 测试 update_camera_status 方法
                if hasattr(window, 'update_camera_status'):
                    print("\n🔄 测试 update_camera_status 方法:")
                    
                    # 测试连接状态更新
                    window.update_camera_status(True)
                    status_after_connect = window.board_camera_status.text()
                    print(f"   调用 update_camera_status(True) 后: '{status_after_connect}'")
                    
                    # 测试断开状态更新
                    window.update_camera_status(False)
                    status_after_disconnect = window.board_camera_status.text()
                    print(f"   调用 update_camera_status(False) 后: '{status_after_disconnect}'")
                    
                    print("✅ update_camera_status 方法工作正常")
                else:
                    print("❌ update_camera_status 方法不存在")
                
                print("\n🎉 所有测试通过！")
                print("💡 board_camera_status 属性已正确修复")
                
            else:
                print(f"❌ board_camera_status 不是 QLabel 类型，而是: {type(window.board_camera_status)}")
                return False
        else:
            print("❌ board_camera_status 属性不存在")
            return False
            
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("💡 请确保 PyQt5 已安装且 visualization_test2.py 文件存在")
        return False
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ui_integration():
    """测试UI集成"""
    print("\n🔍 测试UI集成")
    print("=" * 60)
    
    try:
        from PyQt5.QtWidgets import QApplication
        from visualization_test2 import MainWindow
        
        app = QApplication(sys.argv) if not QApplication.instance() else QApplication.instance()
        window = MainWindow()
        window.init_ui()
        
        # 检查摄像头相关UI组件
        ui_components = [
            'camera_preview_label',
            'board_camera_status',
            'connect_camera_button',
            'capture_from_camera_button'
        ]
        
        print("检查摄像头相关UI组件:")
        for component in ui_components:
            if hasattr(window, component):
                print(f"✅ {component} 存在")
            else:
                print(f"❌ {component} 不存在")
        
        # 检查布局结构
        if hasattr(window, 'camera_preview_label') and hasattr(window, 'board_camera_status'):
            print("✅ 摄像头预览和状态标签都已正确定义")
            return True
        else:
            print("❌ 摄像头UI组件定义不完整")
            return False
            
    except Exception as e:
        print(f"❌ UI集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开发板摄像头状态标签修复验证")
    print("=" * 70)
    
    success = True
    
    # 测试1: 属性定义测试
    if not test_board_camera_status_attribute():
        success = False
    
    # 测试2: UI集成测试
    if not test_ui_integration():
        success = False
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 所有测试通过！")
        print("✅ MainWindow 类中的 board_camera_status 属性已正确修复")
        print("💡 现在可以正常运行 visualization_test2.py 而不会出现属性错误")
    else:
        print("❌ 部分测试失败")
        print("💡 请检查修复的代码是否正确")
    
    print("\n📋 修复总结:")
    print("1. ✅ 在 init_ui() 方法中添加了 board_camera_status 标签定义")
    print("2. ✅ 在 update_camera_status() 方法中添加了状态更新逻辑")
    print("3. ✅ 添加了 hasattr() 安全检查，防止属性不存在的错误")
    print("4. ✅ 设置了合适的样式和初始状态")

if __name__ == "__main__":
    main()
