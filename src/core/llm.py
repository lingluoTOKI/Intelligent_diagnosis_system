"""DeepSeek requests, bounded conversation state, and explicit response provenance."""
from dataclasses import dataclass
import os
import threading
import time
from urllib.parse import urlsplit
import requests

from configs.runtime_config import load_app_config
from .classification import valid_prediction
from .reference_advice import get_reference_advice


@dataclass(frozen=True)
class AdviceResult:
    success: bool
    content: str
    source: str
    error: str = None


class DeepSeekAPI:
    def __init__(self, api_key=None, config=None, post=None, sleep=None):
        self.config = config or load_app_config()
        self.api_key = api_key if api_key is not None else os.getenv("DEEPSEEK_API_KEY", "")
        self.endpoint = self.config.api_endpoint
        self.model = self.config.api_model
        self._post = post or requests.post
        self._sleep = sleep or time.sleep
        self._lock = threading.RLock()
        self._history = []
        self.last_response = None

    def set_api_key(self, api_key):
        with self._lock:
            self.api_key = api_key or ""
            self._history.clear()

    def reset_conversation(self):
        with self._lock:
            self._history.clear()

    def _request(self, messages):
        if not self.api_key or not self.api_key.strip():
            return AdviceResult(False, "请先配置DeepSeek API密钥。", "error", "missing_api_key")
        last_error = "AI服务暂时不可用"
        for attempt in range(self.config.api_attempts):
            try:
                response = self._post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {self.api_key.strip()}", "Content-Type": "application/json"},
                    json={"model": self.model, "messages": messages, "temperature": 0.3,
                          "max_tokens": 2000, "stream": False},
                    timeout=(self.config.api_connect_timeout, self.config.api_read_timeout),
                )
                if response.status_code == 200:
                    content = response.json()["choices"][0]["message"]["content"]
                    if not isinstance(content, str) or not content.strip():
                        raise ValueError("API返回空内容")
                    return AdviceResult(True, content.strip(), "deepseek")
                last_error = f"API请求失败，HTTP {response.status_code}"
                if response.status_code not in {429, 500, 502, 503, 504}:
                    break
            except (requests.Timeout, requests.ConnectionError) as error:
                last_error = f"AI服务连接失败：{type(error).__name__}"
            except (ValueError, KeyError, IndexError, TypeError):
                last_error = "API响应格式无效"
                break
            except requests.RequestException as error:
                last_error = f"AI请求失败：{type(error).__name__}"
                break
            if attempt + 1 < self.config.api_attempts:
                self._sleep(min(0.5 * (2 ** attempt), 2))
        return AdviceResult(False, f"AI服务不可用：{last_error}。请稍后重试。", "error", last_error)

    def request_custom_advice(self, prompt):
        with self._lock:
            if not isinstance(prompt, str) or len(prompt.strip()) < 2:
                self.last_response = AdviceResult(False, "请输入有效的问题。", "error", "invalid_prompt")
                return self.last_response
            text = prompt.strip()[:4000]
            system = {"role": "system", "content": "你是眼科辅助分析助手。分类输出仅为模型预测，不能确诊。提供清楚的参考信息，避免确定性诊断；症状与图像结论需由专业医生复核。"}
            messages = [system] + self._history + [{"role": "user", "content": text}]
            self.last_response = self._request(messages)
            if self.last_response.success:
                self._history += [{"role": "user", "content": text},
                                  {"role": "assistant", "content": self.last_response.content}]
                self._history = self._history[-2 * self.config.chat_turns:]
            return self.last_response

    def get_custom_advice(self, prompt):
        return self.request_custom_advice(prompt).content

    def request_treatment_advice(self, disease_name, confidence):
        if not valid_prediction(disease_name, confidence):
            raise ValueError("没有有效分类结果，不能生成分类辅助建议")
        with self._lock:
            result = self._request([
                {"role": "system", "content": "你是眼科辅助分析助手。以下是图像分类模型的预测，不是临床诊断。请说明结果局限、一般参考信息和专业复核建议，不据此确诊或开处方。"},
                {"role": "user", "content": f"预测类别：{disease_name}；模型输出概率：{float(confidence):.4f}。请提供辅助参考说明。"},
            ])
            if not result.success:
                result = AdviceResult(False, result.content + "\n\n" + self._get_default_advice(disease_name),
                                      "local_reference", result.error)
            self.last_response = result
            return result

    def get_treatment_advice(self, disease_name, confidence):
        return self.request_treatment_advice(disease_name, confidence).content

    def _get_default_advice(self, disease_name):
        return "【本地参考资料，非DeepSeek在线回答】\n" + get_reference_advice(disease_name)

    def test_network_connection(self):
        origin = urlsplit(self.endpoint)
        try:
            response = requests.get(f"{origin.scheme}://{origin.netloc}",
                                    timeout=self.config.api_connect_timeout)
            return response.status_code < 500, f"API主机响应HTTP {response.status_code}（不代表密钥验证成功）"
        except requests.RequestException:
            return False, "无法连接API主机"


class MedicalAIService(DeepSeekAPI):
    """Compatibility name for existing text/voice callers; uses the same request service."""
