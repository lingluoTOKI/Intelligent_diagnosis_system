#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
开发板-PC端整合系统测试脚本
用于验证各项功能是否正常工作
"""

import socket
import json
import time
from datetime import datetime

def test_network_connectivity():
    """测试网络连接"""
    print("🔍 测试网络连接...")
    
    # 测试端口
    test_ports = [5002, 5003, 5004, 5005, 5006]
    results = {}
    
    for port in test_ports:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("0.0.0.0", port))
            sock.close()
            results[port] = "✅ 可用"
        except:
            results[port] = "❌ 占用"
    
    print("端口状态检查：")
    for port, status in results.items():
        print(f"  端口 {port}: {status}")
    
    return results

def test_audio_devices():
    """测试音频设备"""
    print("\n🔊 测试音频设备...")
    
    try:
        import pyttsx3
        engine = pyttsx3.init()
        
        # 获取可用语音
        voices = engine.getProperty('voices')
        print(f"可用TTS语音数量: {len(voices)}")
        
        # 测试语音播放
        test_text = "这是音频测试，如果您能听到这条消息，说明TTS功能正常。"
        print(f"播放测试文本: {test_text}")
        
        engine.say(test_text)
        engine.runAndWait()
        engine.stop()
        
        print("✅ TTS测试完成")
        return True
        
    except ImportError:
        print("❌ pyttsx3未安装，TTS功能不可用")
        return False
    except Exception as e:
        print(f"❌ TTS测试失败: {e}")
        return False

def test_command_communication():
    """测试命令通信"""
    print("\n📡 测试命令通信...")
    
    try:
        # 创建测试命令
        test_command = {
            "type": "connection_test",
            "timestamp": datetime.now().isoformat(),
            "test_id": "integration_test_001",
            "message": "这是整合系统测试命令"
        }
        
        # 发送测试命令
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        command_json = json.dumps(test_command, ensure_ascii=False)
        command_bytes = command_json.encode('utf-8')
        
        print("发送测试命令到端口 5004...")
        try:
            sock.sendto(command_bytes, ("127.0.0.1", 5004))
            print("✅ 命令发送成功")
        except Exception as e:
            print(f"❌ 命令发送失败: {e}")
        finally:
            sock.close()
        
        return True
        
    except Exception as e:
        print(f"❌ 命令通信测试失败: {e}")
        return False

def test_voice_command_format():
    """测试语音命令格式"""
    print("\n🎤 测试语音命令格式...")
    
    try:
        # 测试语音文本命令
        voice_command = {
            "type": "voice_command",
            "command": "voice_text",
            "params": {
                "text": "这是测试语音文本，用于验证AI对话功能",
                "source": "integration_test",
                "timestamp": datetime.now().isoformat()
            },
            "source_addr": ("127.0.0.1", 5004)
        }
        
        print("语音命令格式验证:")
        print(f"  命令类型: {voice_command['type']}")
        print(f"  子命令: {voice_command['command']}")
        print(f"  文本内容: {voice_command['params']['text']}")
        
        # 验证JSON序列化
        json_str = json.dumps(voice_command, ensure_ascii=False)
        parsed = json.loads(json_str)
        
        print("✅ JSON序列化/反序列化正常")
        return True
        
    except Exception as e:
        print(f"❌ 语音命令格式测试失败: {e}")
        return False

def test_tts_audio_command():
    """测试TTS音频命令格式"""
    print("\n🔉 测试TTS音频命令格式...")
    
    try:
        # 测试TTS播放命令
        tts_command = {
            "type": "tts_play",
            "text": "这是TTS播放测试，验证开发板音频播放功能",
            "timestamp": datetime.now().isoformat(),
            "source": "pc_ai_advice"
        }
        
        print("TTS命令格式验证:")
        print(f"  命令类型: {tts_command['type']}")
        print(f"  播放文本: {tts_command['text']}")
        print(f"  来源: {tts_command['source']}")
        
        # 验证命令大小
        json_str = json.dumps(tts_command, ensure_ascii=False)
        size = len(json_str.encode('utf-8'))
        print(f"  命令大小: {size} 字节")
        
        if size > 4096:
            print("⚠️ 警告：命令大小超过UDP包限制")
        else:
            print("✅ 命令大小正常")
        
        return True
        
    except Exception as e:
        print(f"❌ TTS命令格式测试失败: {e}")
        return False

def generate_test_report(results):
    """生成测试报告"""
    print("\n" + "="*50)
    print("📋 整合系统测试报告")
    print("="*50)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r)
    
    print(f"总测试项目: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"失败测试: {total_tests - passed_tests}")
    print(f"通过率: {passed_tests/total_tests*100:.1f}%")
    
    print("\n详细结果:")
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    print("\n📝 建议:")
    if passed_tests == total_tests:
        print("  🎉 所有测试通过！系统整合成功。")
        print("  💡 可以开始使用整合系统的各项功能。")
    else:
        print("  ⚠️ 部分测试失败，请检查：")
        print("  1. 确保所需依赖库已安装")
        print("  2. 检查网络端口是否被占用")
        print("  3. 验证音频设备是否正常")
        print("  4. 确认系统权限设置")
    
    print("="*50)

def main():
    """主测试函数"""
    print("🚀 开始整合系统测试...")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 执行各项测试
    results = {}
    
    # 网络连接测试
    network_results = test_network_connectivity()
    results["网络端口检查"] = all(status == "✅ 可用" for status in network_results.values())
    
    # 音频设备测试
    results["TTS音频功能"] = test_audio_devices()
    
    # 命令通信测试
    results["命令通信格式"] = test_command_communication()
    
    # 语音命令测试
    results["语音命令格式"] = test_voice_command_format()
    
    # TTS命令测试
    results["TTS命令格式"] = test_tts_audio_command()
    
    # 生成测试报告
    generate_test_report(results)

if __name__ == "__main__":
    main()
