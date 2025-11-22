# main.py
import os
from tqdm import tqdm

from config.config import config
from core.api_predictor import APIPredictor
from core.local_predictor import LocalPredictor
from utils.data_loader import DataLoader
from utils.file_saver import FileSaver

def create_predictor():
    """创建预测器实例"""
    if config.model.model_type == "deepseek":
        return APIPredictor()
    elif "qwen" in config.model.model_type:
        return LocalPredictor()
    else:
        raise ValueError(f"不支持的模型类型: {config.model.model_type}")

def main():
    """主函数"""
    print("=== 美国人口调查预测系统 ===")
    print(f"模型类型: {config.model.model_type}")
    print(f"输出类型: {config.model.output_type}")
    if "qwen" in config.model.model_type:
        print(f"多GPU: {config.model.use_multi_gpu}")
    
    # 创建预测器
    predictor = create_predictor()
    predictor.initialize()
    
    # 加载数据
    questions = DataLoader.load_questions()
    
    # 文件保存器
    file_saver = FileSaver(predictor)
    
    # 开始预测
    results = {}
    print("开始预测...")
    
    for question in tqdm(questions, desc="处理问题"):
        try:
            qid = question['input_id']
            q_raw = question['question_raw']
            mapping = question['mapping']
            
            # 预测分布
            dist = predictor.predict_one(q_raw, mapping)
            results[qid] = dist
            
            # 打印第一个问题的结果作为示例
            if len(results) == 1:
                print(f"\n第一个问题预测结果示例:")
                print(f"问题ID: {qid}")
                print(f"问题Key: {question['qkey']}")
                print(f"问题: {q_raw}")
                print(f"预测分布: {dist}")
                
                if config.model.output_type == "logits":
                    prob_dist = predictor.logits_to_probability(dist)
                    print(f"转换后的概率分布: {prob_dist}")
                    print(f"概率总和: {sum(prob_dist.values()):.4f}")
                else:
                    print(f"概率总和: {sum(dist.values()):.4f}")
                
        except Exception as e:
            print(f"处理问题 {question.get('input_id', 'unknown')} 时出错: {e}")
            continue
    
    # 保存结果
    file_saver.save_results(results, questions)
    
    # 打印统计信息
    output_files = file_saver.get_output_files()
    print(f"\n=== 预测完成 ===")
    print(f"总问题数: {len(results)}")
    print(f"输出文件:")
    for file_type, file_path in output_files.items():
        if os.path.exists(file_path):
            print(f"  - {file_type.upper()}: {file_path}")

if __name__ == "__main__":
    # 在这里修改配置即可切换不同模型和输出类型
    # config.update_model_config(model_type="deepseek", output_type="logits")  # Deepseek API + logits
    # config.update_model_config(model_type="qwen7b", output_type="probability")  # Qwen 7B + 概率
    config.update_model_config(model_type="qwen32b", output_type="logits", use_multi_gpu=True)  # Qwen 32B + logits + 多GPU
    
    main()
