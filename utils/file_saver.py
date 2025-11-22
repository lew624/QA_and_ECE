# file_saver.py
import json
import pandas as pd
import os
from typing import Dict
import numpy as np

from config.config import config
from core.base_predictor import BasePredictor

class FileSaver:
    """文件保存器"""
    
    def __init__(self, predictor: BasePredictor):
        self.predictor = predictor
        self.output_paths = config.get_output_paths()
        
        # 创建输出目录
        os.makedirs(config.output.output_dir, exist_ok=True)
    
    def save_results(self, results: Dict, original_questions: list):
        """保存所有结果"""
        if config.output.save_json:
            self._save_json(results)
        
        if config.output.save_excel:
            self._save_excel(results, original_questions)
        
        if config.output.save_csv:
            self._save_csv(results, original_questions)
    
    def _save_json(self, results: Dict):
        """保存JSON格式"""
        with open(self.output_paths["json"], "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"JSON保存完成 -> {self.output_paths['json']}")
    
    def _save_excel(self, results: Dict, original_questions: list):
        """保存Excel格式"""
        rows = []
        for qid, dist in results.items():
            # 找到原始问题数据
            question_data = next((q for q in original_questions if q['input_id'] == qid), {})
            
            row = {
                'input_id': qid,
                'qkey': question_data.get('qkey', ''),
                'question_raw': question_data.get('question_raw', '')
            }
            
            # 保存主要输出
            if config.model.output_type == "logits":
                for option, logit in dist.items():
                    row[f'logit_{option}'] = logit
                
                # 同时保存转换后的概率（用于兼容性）
                prob_dist = self.predictor.logits_to_probability(dist)
                for option, prob in prob_dist.items():
                    row[f'pred_{option}'] = prob
            else:
                for option, prob in dist.items():
                    row[f'pred_{option}'] = prob
            
            rows.append(row)
        
        df_output = pd.DataFrame(rows)
        df_output.to_excel(self.output_paths["excel"], index=False, engine='openpyxl')
        print(f"Excel保存完成 -> {self.output_paths['excel']}")
    
    def _save_csv(self, results: Dict, original_questions: list):
        """保存CSV格式"""
        rows = []
        for qid, dist in results.items():
            # 找到原始问题数据
            question_data = next((q for q in original_questions if q['input_id'] == qid), {})
            
            row = {
                'input_id': qid,
                'qkey': question_data.get('qkey', ''),
                'question_raw': question_data.get('question_raw', '')
            }
            
            # 保存主要输出
            if config.model.output_type == "logits":
                for option, logit in dist.items():
                    row[f'logit_{option}'] = logit
                
                # 同时保存转换后的概率（用于兼容性）
                prob_dist = self.predictor.logits_to_probability(dist)
                for option, prob in prob_dist.items():
                    row[f'pred_{option}'] = prob
            else:
                for option, prob in dist.items():
                    row[f'pred_{option}'] = prob
            
            rows.append(row)
        
        df_output = pd.DataFrame(rows)
        df_output.to_csv(self.output_paths["csv"], index=False)
        print(f"CSV保存完成 -> {self.output_paths['csv']}")
    
    def get_output_files(self) -> Dict[str, str]:
        """获取输出文件路径"""
        return self.output_paths
