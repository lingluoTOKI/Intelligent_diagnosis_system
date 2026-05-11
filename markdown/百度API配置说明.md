# 🔧 百度语音API配置说明

## 📋 问题描述

当前系统显示以下错误：
```
[ERROR] 百度API请求失败: 401 - {"error_description":"unknown client id","error":"invalid_client"}
```

这个错误表示百度语音API的密钥配置有问题。

## 🔑 解决方案

### 1. 获取百度语音API密钥

1. **访问百度AI开放平台**
   - 网址：https://ai.baidu.com/
   - 注册/登录百度账号

2. **创建语音应用**
   - 进入"语音技术" → "语音识别"
   - 点击"立即使用"
   - 创建新应用

3. **获取API密钥**
   - 在应用详情页面找到：
     - **API Key** (应用ID)
     - **Secret Key** (应用密钥)

### 2. 配置API密钥

#### 方法一：在程序中配置
在 `visualization_test2.py` 中找到以下代码：

```python
# 在SmartVoiceManager类的init_voice_components方法中
self.baidu_api = BaiduSpeechAPI("your_api_key", "your_secret_key")
```

将 `"your_api_key"` 和 `"your_secret_key"` 替换为您的实际密钥。

#### 方法二：使用配置文件
创建一个 `api_config.json` 文件：

```json
{
    "baidu_api_key": "您的百度API Key",
    "baidu_secret_key": "您的百度Secret Key",
    "deepseek_api_key": "您的DeepSeek API Key"
}
```

### 3. 验证配置

配置完成后，重新运行程序：
```bash
python visualization_test2.py
```

## 🎯 替代方案

如果暂时无法配置百度API，系统会自动使用本地语音识别：

1. **本地识别模式**
   - 无需网络连接
   - 使用本地语音识别引擎
   - 准确率相对较低但可用

2. **Google语音识别**
   - 系统会自动尝试Google API
   - 需要网络连接
   - 准确率较高

## 📝 配置步骤详解

### 步骤1：注册百度AI平台
1. 访问 https://ai.baidu.com/
2. 使用百度账号登录
3. 完成实名认证（如需要）

### 步骤2：创建语音应用
1. 进入"语音技术" → "语音识别"
2. 点击"立即使用"
3. 填写应用信息：
   - 应用名称：眼科诊断系统
   - 应用描述：AI眼科疾病诊断系统语音识别
   - 选择"语音识别"服务

### 步骤3：获取密钥
1. 创建完成后，进入应用详情
2. 复制以下信息：
   - **API Key** (应用ID)
   - **Secret Key** (应用密钥)

### 步骤4：更新代码
找到代码中的这一行：
```python
self.baidu_api = BaiduSpeechAPI("your_api_key", "your_secret_key")
```

替换为：
```python
self.baidu_api = BaiduSpeechAPI("实际的API Key", "实际的Secret Key")
```

## ⚠️ 注意事项

1. **密钥安全**
   - 不要将API密钥提交到公开代码库
   - 建议使用环境变量或配置文件

2. **配额限制**
   - 免费版有调用次数限制
   - 注意监控使用量

3. **网络要求**
   - 需要稳定的网络连接
   - 建议使用国内网络

## 🔄 故障排除

### 常见错误及解决方案

1. **401错误 - invalid_client**
   - 检查API Key和Secret Key是否正确
   - 确认应用是否已启用语音识别服务

2. **403错误 - access denied**
   - 检查应用权限设置
   - 确认服务是否在可用区域

3. **网络连接错误**
   - 检查网络连接
   - 尝试使用VPN或代理

## 📞 技术支持

如果遇到配置问题，可以：

1. **查看百度AI文档**
   - https://ai.baidu.com/ai-doc/SPEECH/

2. **联系百度技术支持**
   - 在百度AI平台提交工单

3. **使用本地识别**
   - 系统会自动降级到本地识别模式

---

*配置完成后，语音识别功能将正常工作，提供更好的用户体验！* 