#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
语音功能诊断脚本
帮助用户快速诊断语音识别问题
"""

import sys
import time
import traceback

def test_imports():
    """测试必要库的导入"""
    print("="*50)
    print("步骤 1: 测试库导入")
    print("="*50)
    
    success = True
    
    try:
        import speech_recognition as sr
        print("[OK] speech_recognition 导入成功")
    except Exception as e:
        print(f"[FAIL] speech_recognition 导入失败: {e}")
        success = False
    
    try:
        import pyttsx3
        print("[OK] pyttsx3 导入成功")
    except Exception as e:
        print(f"[FAIL] pyttsx3 导入失败: {e}")
        success = False
    
    try:
        import pyaudio
        print("[OK] pyaudio 导入成功")
    except Exception as e:
        print(f"[FAIL] pyaudio 导入失败: {e}")
        success = False
    
    return success

def test_audio_devices():
    """测试音频设备"""
    print("\n" + "="*50)
    print("步骤 2: 测试音频设备")
    print("="*50)
    
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        
        print(f"音频驱动: {p.get_host_api_count()} 个主机API")
        
        input_devices = []
        output_devices = []
        
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                input_devices.append((i, info['name']))
            if info['maxOutputChannels'] > 0:
                output_devices.append((i, info['name']))
        
        print(f"\n输入设备 ({len(input_devices)} 个):")
        for idx, name in input_devices:
            print(f"  [{idx}] {name}")
            
        print(f"\n输出设备 ({len(output_devices)} 个):")
        for idx, name in output_devices:
            print(f"  [{idx}] {name}")
        
        p.terminate()
        
        if not input_devices:
            print("[WARNING] 未找到输入设备！")
            return False
            
        return True
        
    except Exception as e:
        print(f"[FAIL] 音频设备测试失败: {e}")
        return False

def test_microphone():
    """测试麦克风访问"""
    print("\n" + "="*50)
    print("步骤 3: 测试麦克风访问")
    print("="*50)
    
    try:
        import speech_recognition as sr
        
        r = sr.Recognizer()
        mic = sr.Microphone()
        
        print("正在测试麦克风访问...")
        with mic as source:
            print("[OK] 麦克风访问成功")
            print("正在调整环境噪音...")
            r.adjust_for_ambient_noise(source, duration=1)
            print("[OK] 环境噪音调整完成")
            
        return True
        
    except Exception as e:
        print(f"[FAIL] 麦克风测试失败: {e}")
        print("可能的原因:")
        print("1. 麦克风未连接或已被其他应用占用")
        print("2. 麦克风权限未开启")
        print("3. 音频驱动程序问题")
        return False

def test_speech_recognition():
    """测试语音识别（不实际录音）"""
    print("\n" + "="*50)
    print("步骤 4: 测试语音识别服务")
    print("="*50)
    
    try:
        import speech_recognition as sr
        from io import BytesIO
        
        r = sr.Recognizer()
        
        # 使用一个简单的测试音频数据（空的，只是测试API是否可达）
        print("正在测试 Google Speech API 连接...")
        
        # 这里我们不实际调用 recognize_google 因为需要真实音频
        # 但可以测试网络连接
        import requests
        response = requests.get("https://www.google.com", timeout=5)
        if response.status_code == 200:
            print("[OK] 网络连接正常")
        else:
            print("[WARNING] 网络连接可能有问题")
            
        print("[INFO] Google Speech API 需要实际音频数据才能测试")
        print("[INFO] 请在应用中尝试语音输入以完成测试")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] 网络连接测试失败: {e}")
        print("可能的原因:")
        print("1. 网络连接问题")
        print("2. 防火墙阻止连接")
        print("3. 代理设置问题")
        return False

def test_tts():
    """测试文字转语音"""
    print("\n" + "="*50)
    print("步骤 5: 测试文字转语音")
    print("="*50)
    
    try:
        import pyttsx3
        
        engine = pyttsx3.init()
        print("[OK] TTS引擎初始化成功")
        
        voices = engine.getProperty('voices')
        print(f"[INFO] 可用语音数量: {len(voices) if voices else 0}")
        
        if voices:
            for i, voice in enumerate(voices[:3]):  # 只显示前3个
                print(f"  语音 {i}: {voice.name} ({voice.id})")
        
        # 测试语音参数设置
        engine.setProperty('rate', 180)
        engine.setProperty('volume', 0.8)
        print("[OK] TTS参数设置成功")
        
        print("[INFO] 如需测试语音播放，请在应用中启用语音回复功能")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] TTS测试失败: {e}")
        return False

def main():
    """主诊断流程"""
    print("AI眼科疾病智诊系统 - 语音功能诊断工具")
    print("="*60)
    
    tests = [
        ("库导入测试", test_imports),
        ("音频设备测试", test_audio_devices),
        ("麦克风测试", test_microphone),
        ("语音识别服务测试", test_speech_recognition),
        ("文字转语音测试", test_tts)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[ERROR] {test_name} 执行异常: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # 总结
    print("\n" + "="*60)
    print("诊断结果总结")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n通过率: {passed}/{len(results)} ({passed/len(results)*100:.1f}%)")
    
    if passed == len(results):
        print("\n[SUCCESS] 所有测试通过！语音功能应该可以正常使用。")
        print("如果仍有问题，请检查:")
        print("1. 在应用中是否勾选了'启用语音对话'")
        print("2. 说话时是否靠近麦克风")
        print("3. 说话是否清晰（支持中文和英文）")
    else:
        print(f"\n[WARNING] {len(results)-passed} 个测试失败，语音功能可能无法正常工作。")
        print("建议操作:")
        print("1. 运行 'pip install SpeechRecognition pyttsx3 pyaudio' 安装依赖")
        print("2. 检查麦克风连接和权限设置")
        print("3. 确保网络连接正常")
        print("4. 重启应用程序")
    
    print("="*60)

if __name__ == "__main__":
    main()
    input("\n按回车键退出...")