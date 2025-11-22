# deepseek_api_predictor.py
import requests
import time
from typing import Dict, List
from core.base_predictor import BasePredictor
from config.config import config

class APIPredictor(BasePredictor):
    """API预测器"""
    
    def __init__(self):
        super().__init__()
        self.max_retries = 3
    
    def initialize(self):
        """初始化API连接"""
        if config.model.api_key == "your_api_key_here":
            raise ValueError("请设置你的API密钥")
        print("API预测器初始化完成")
    
    def call_api(self, messages: List[Dict]) -> str:
        """调用API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.model.api_key}"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": config.model.temperature,
            "max_tokens": config.model.max_tokens,
            "stream": False
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(config.model.api_url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                
                result = response.json()
                return result["choices"][0]["message"]["content"]
                
            except requests.exceptions.RequestException as e:
                print(f"API调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    raise Exception(f"API调用失败，已达最大重试次数: {e}")
    
    def predict_one(self, question_raw: str, mapping: Dict[str, str]) -> Dict[str, float]:
        """预测单个问题的分布"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.build_user_prompt(question_raw, mapping)}
        ]
        
        try:
            # 调用API
            reply = self.call_api(messages)
            return self.parse_response(reply, mapping)
            
        except Exception as e:
            print(f"预测过程中出错: {e}")
            # 返回默认分布
            if config.model.output_type == "logits":
                default_dist = {k: 0.0 for k in mapping}
            else:
                default_dist = {k: 1.0 for k in mapping}
            
            return self.remove_refused_options(default_dist, mapping)
