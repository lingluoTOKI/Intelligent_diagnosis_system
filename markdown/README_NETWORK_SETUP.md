# 🏥 智能医疗诊断系统 - 网络连接指南

## 🔧 解决连接问题

### 问题描述
开发板无法连接到PC端，显示"连接测试超时"错误。

### 📋 快速解决步骤

#### 方法1: 使用快速启动脚本（推荐）
```bash
python quick_start.py
```
选择"智能模式"，系统会自动检测并配置网络。

#### 方法2: 手动配置IP地址

1. **获取PC端IP地址**
   ```bash
   # Windows
   ipconfig
   
   # Linux/Mac
   ifconfig
   # 或
   ip addr show
   ```

2. **在开发板端配置IP**
   - 启动开发板系统后，按 `i` 键
   - 输入正确的PC端IP地址
   - 系统会自动测试连接

#### 方法3: 修改配置文件
创建 `network_config.json` 文件：
```json
{
  "pc_ip": "192.168.1.100",
  "timestamp": "2024-01-01T12:00:00",
  "board_id": "medical_board_001"
}
```

### 🌐 网络配置要求

#### 端口配置
- **5002**: 摄像头数据传输
- **5003**: 诊断结果接收  
- **5004**: 命令控制通信
- **5005-5007**: 语音数据传输
- **5008**: 触摸控制

#### 防火墙设置
确保以下端口允许通信：
```bash
# Windows防火墙
netsh advfirewall firewall add rule name="Medical System" dir=in action=allow protocol=UDP localport=5002-5008

# Linux iptables
sudo iptables -A INPUT -p udp --dport 5002:5008 -j ACCEPT
```

### 🚀 启动顺序

#### 推荐启动顺序：
1. **首先启动PC端**
   ```bash
   python visualization_test2.py
   ```

2. **然后启动开发板端**
   ```bash
   python board_integrated_system.py
   ```

#### 或使用系统启动器：
```bash
python system_launcher.py
# 选择"双端模式"
```

### 🔍 故障排除

#### 常见问题及解决方案

1. **"连接测试超时"**
   - ✅ 确保PC端已启动
   - ✅ 检查IP地址是否正确
   - ✅ 确认两设备在同一网络

2. **"端口被占用"**
   ```bash
   # 检查端口占用
   netstat -an | grep 5004
   
   # 杀死占用进程
   sudo fuser -k 5004/udp
   ```

3. **"防火墙阻止"**
   - 临时关闭防火墙测试
   - 或添加端口例外规则

4. **网络不通**
   ```bash
   # 测试基本连通性
   ping PC_IP_ADDRESS
   
   # 测试端口连通性
   telnet PC_IP_ADDRESS 5004
   ```

### 📱 开发板控制台命令

在开发板控制台模式下可用的命令：

- `c` - 启动摄像头预览
- `p` - 拍照并诊断  
- `r` - 开始录音
- `s` - 显示系统状态
- `n` - 测试网络连接
- `i` - 设置PC端IP地址
- `q` - 退出系统

### 🔧 高级配置

#### 自定义网络参数
修改 `board_integrated_system.py` 中的配置：
```python
PC_IP = "你的PC端IP地址"
COMMAND_PORT = 5004  # 命令端口
CAMERA_PORT = 5002   # 摄像头端口
```

#### 创建统一配置文件
创建 `system_config.py`：
```python
PC_IP = "192.168.1.100"
CAMERA_PORT = 5002
DIAGNOSIS_PORT = 5003
COMMAND_PORT = 5004
# ... 其他配置
```

### 📞 技术支持

如果问题仍然存在：

1. 检查系统日志获取详细错误信息
2. 确认网络环境和设备配置
3. 尝试使用不同的网络或IP地址
4. 检查Python环境和依赖库

### 🌟 成功连接标志

当连接成功时，你会看到：
```
✅ [网络] 连接成功! PC端IP: 192.168.1.100, 延迟: 5ms
[成功] 系统初始化完成
[配置] PC端IP: 192.168.1.100
```

---

*最后更新: 2024年*






