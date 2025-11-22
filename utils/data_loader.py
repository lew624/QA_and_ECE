# data_loader.py
import pandas as pd
import ast
from typing import List, Dict, Any
from config.config import config

class DataLoader:
    """数据加载器"""
    
    @staticmethod
    def load_questions() -> List[Dict[str, Any]]:
        """加载问题数据"""
        print("正在读取问题数据...")
        df = pd.read_excel(config.data.excel_path, sheet_name=config.data.sheet_name, engine="openpyxl")
        
        questions = []
        for _, row in df.iterrows():
            try:
                questions.append({
                    'input_id': row["input_id"],
                    'qkey': row["qkey"],
                    'question_raw': row["question_raw"],
                    'mapping': ast.literal_eval(row["mapping"])
                })
            except Exception as e:
                print(f"解析问题 {row.get('input_id', 'unknown')} 时出错: {e}")
                continue
        
        print(f"成功加载 {len(questions)} 个问题")
        return questions
