# qwen7b_predictor.py
import json
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np

from config.config import config

class BasePredictor(ABC):
    """基础预测器抽象类"""
    
    def __init__(self):
        self.system_prompt = self._build_system_prompt()
        self.refused_keywords = ['refused', 'Refused', 'REFUSED', '拒绝', '不知道', '不清楚', 'Not sure', 'not sure']
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        if config.model.output_type == "logits":
            return """你是一个专门分析美国人口调查数据的AI助手。你需要根据问题内容，预测美国全部人口选择各个选项的logits分布。

请严格按照以下要求回答：
1. 分析每个选项在美国人口中可能的选择倾向
2. 考虑美国的社会文化背景、人口统计学特征
3. 基于常识和典型调查数据模式进行合理推断
4. 输出格式必须是有效的JSON对象，键为选项字母（A、B、C等），值为对应的logits（可以是任意实数，正数、负数或零）
5. logits表示每个选项的相对可能性，数值越大表示该选项越可能被选择
6. 不要对logits进行归一化，不要使用softmax
7. 不要包含任何额外的文本解释，只输出JSON

请准确反映美国人口的真实选择倾向。"""
        else:
            return """你是一个专门分析美国人口调查数据的AI助手。你需要根据问题内容，预测美国全部人口选择各个选项的概率分布。

请严格按照以下要求回答：
1. 分析每个选项在美国人口中可能的选择倾向
2. 考虑美国的社会文化背景、人口统计学特征
3. 基于常识和典型调查数据模式进行合理推断
4. 输出格式必须是有效的JSON对象，键为选项字母（A、B、C等），值为对应的概率（0-1之间的小数）
5. 所有选项的概率总和必须为1
6. 不要包含任何额外的文本解释，只输出JSON

请准确反映美国人口的真实选择倾向。"""
    
    def build_user_prompt(self, question_raw: str, mapping: Dict[str, str]) -> str:
        """构建用户提示词"""
        options_text = "\n".join([f"{key}: {value}" for key, value in mapping.items()])
        
        if config.model.output_type == "logits":
            example = '{"A": 2.5, "B": 1.8, "C": -0.3, "D": -1.2}'
            output_type = "logits分布"
        else:
            example = '{"A": 0.35, "B": 0.45, "C": 0.15, "D": 0.05}'
            output_type = "概率分布"
        
        prompt = f"""请分析以下调查问题，并预测美国全部人口选择各个选项的{output_type}：

问题: {question_raw}

选项:
{options_text}

请输出JSON格式的{output_type}，例如：{example}

{output_type}："""
        return prompt
    
    def remove_refused_options(self, dist: Dict[str, float], mapping: Dict[str, str]) -> Dict[str, float]:
        """处理拒绝选项"""
        all_options = list(mapping.keys())
        
        # 检查是否有明显的拒绝选项
        has_refused = False
        for option_key, option_text in mapping.items():
            option_text_lower = option_text.lower()
            if any(keyword.lower() in option_text_lower for keyword in self.refused_keywords):
                has_refused = True
                break
        
        if has_refused:
            # 过滤掉包含拒绝关键词的选项
            filtered_options = []
            for option in all_options:
                option_text = mapping.get(option, '').lower()
                if not any(keyword.lower() in option_text for keyword in self.refused_keywords):
                    filtered_options.append(option)
        else:
            filtered_options = all_options
        
        # 创建新的分布
        new_dist = {}
        for option in filtered_options:
            if option in dist:
                new_dist[option] = dist[option]
            else:
                new_dist[option] = 0.0 if config.model.output_type == "logits" else 0.0
        
        # 对于概率输出，需要重新归一化
        if config.model.output_type == "probability" and new_dist:
            total = sum(new_dist.values())
            if total > 0:
                new_dist = {k: v/total for k, v in new_dist.items()}
            else:
                uniform_prob = 1.0 / len(filtered_options)
                new_dist = {k: uniform_prob for k in filtered_options}
        
        return new_dist
    
    def extract_json_from_text(self, text: str) -> str:
        """从文本中提取JSON字符串"""
        # 尝试找到JSON对象
        json_pattern = r'\{[^{}]*"[^{}]*"[^{}]*\}'
        matches = re.findall(json_pattern, text)
        
        if matches:
            return max(matches, key=len)
        
        # 如果没有找到标准JSON，尝试找到包含数字键值对的部分
        fallback_pattern = r'\{[^{}]*:[^{}]*\}'
        matches = re.findall(fallback_pattern, text)
        if matches:
            return max(matches, key=len)
        
        return None
    
    def logits_to_probability(self, logits_dict: Dict[str, float]) -> Dict[str, float]:
        """将logits转换为概率分布（使用softmax）"""
        if not logits_dict:
            return {}
        
        logits = np.array(list(logits_dict.values()))
        
        # 应用softmax：exp(logits) / sum(exp(logits))
        max_logit = np.max(logits)
        exp_logits = np.exp(logits - max_logit)
        probabilities = exp_logits / np.sum(exp_logits)
        
        # 创建概率字典
        prob_dict = {}
        for i, (option, _) in enumerate(logits_dict.items()):
            prob_dict[option] = float(probabilities[i])
        
        return prob_dict
    
    def parse_response(self, reply: str, mapping: Dict[str, str]) -> Dict[str, float]:
        """解析模型响应"""
        reply = reply.strip()
        
        # 尝试提取JSON
        json_str = self.extract_json_from_text(reply)
        
        if json_str:
            try:
                dist = json.loads(json_str)
                # 确保所有键都是字符串，值都是数字
                dist = {str(k): float(v) for k, v in dist.items()}
                
                # 检查是否包含所有mapping中的选项
                for option in mapping:
                    if option not in dist:
                        dist[option] = 0.0
                
                # 处理拒绝选项
                dist = self.remove_refused_options(dist, mapping)
                return dist
                
            except Exception as e:
                print(f"JSON解析错误: {e}")
                print(f"原始回复: {reply}")
        
        # 如果JSON解析失败，尝试从文本中提取数字
        print(f"无法解析JSON，尝试从文本提取: {reply[:100]}...")
        dist = {}
        for option in mapping:
            if config.model.output_type == "logits":
                pattern = rf'{option}.*?(-?\d+\.?\d*)'
            else:
                pattern = rf'{option}.*?(\d+\.?\d*)'
            
            matches = re.findall(pattern, reply, re.IGNORECASE)
            if matches:
                try:
                    value = float(matches[0])
                    if config.model.output_type == "probability":
                        value = value / 100.0  # 假设是百分比
                    dist[option] = value
                except:
                    dist[option] = 0.0
            else:
                dist[option] = 0.0
        
        # 如果成功提取到一些值
        if any(v != 0 for v in dist.values()):
            dist = self.remove_refused_options(dist, mapping)
            return dist
        
        # 如果所有方法都失败，返回默认分布
        print(f"所有方法失败，使用默认分布")
        if config.model.output_type == "logits":
            default_dist = {k: 0.0 for k in mapping}
        else:
            default_dist = {k: 1.0 for k in mapping}
        
        return self.remove_refused_options(default_dist, mapping)
    
    @abstractmethod
    def predict_one(self, question_raw: str, mapping: Dict[str, str]) -> Dict[str, float]:
        """预测单个问题的分布"""
        pass
    
    @abstractmethod
    def initialize(self):
        """初始化模型"""
        pass
