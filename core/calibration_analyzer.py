# core/calibration_analyzer.py
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import json
import re
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from config.calibration_config import config_manager

class ECEAnalyzer:
    """ECE模型校准分析器 - 专为多选题概率预测设计"""
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.refused_keywords = ['refused', 'Refused', 'REFUSED', '拒绝', '不知道', '不清楚', 'Not sure', 'not sure']
    
    def safe_literal_eval(self, s):
        """安全地解析字符串为字典"""
        if pd.isna(s):
            return {}
        if isinstance(s, dict):
            return s
        if isinstance(s, str):
            try:
                return ast.literal_eval(s)
            except:
                try:
                    s_clean = re.sub(r'(\w+):', r'"\1":', s)
                    return json.loads(s_clean)
                except:
                    return {}
        return {}
    
    def parse_distribution_string(self, dist_str):
        """解析分布字符串"""
        if pd.isna(dist_str):
            return []
        
        if isinstance(dist_str, list):
            return dist_str
        
        if isinstance(dist_str, str):
            try:
                # 清理字符串
                cleaned = dist_str.strip()
                # 移除多余的空白字符
                cleaned = re.sub(r'\s+', ' ', cleaned)
                # 解析为列表
                return ast.literal_eval(cleaned)
            except:
                try:
                    # 尝试其他格式
                    numbers = re.findall(r'[\d.]+', dist_str)
                    return [float(x) for x in numbers]
                except:
                    return []
        
        return []
    
    def parse_distribution(self, dist_str):
        """解析分布字符串为列表"""
        if pd.isna(dist_str):
            return []
        if isinstance(dist_str, list):
            return dist_str
        if isinstance(dist_str, str):
            try:
                # 清理字符串并解析
                cleaned = dist_str.replace('\n', ' ').replace('\t', ' ')
                return ast.literal_eval(cleaned)
            except:
                try:
                    # 尝试其他格式
                    cleaned = re.sub(r'[\[\]\s]+', ' ', dist_str).strip()
                    numbers = [float(x) for x in cleaned.split() if x]
                    return numbers
                except:
                    return []
        return []
    
    def calculate_ece(self, true_probs: List[float], pred_probs: List[float]) -> Dict:
        """计算ECE（Expected Calibration Error）"""
        if len(true_probs) < 2:
            return {'ece': 0, 'details': {}}
        
        # 创建分箱
        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        bin_indices = np.digitize(pred_probs, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        
        ece = 0.0
        bin_details = []
        
        for i in range(self.n_bins):
            mask = bin_indices == i
            bin_size = np.sum(mask)
            
            if bin_size > 0:
                bin_true = np.array(true_probs)[mask]
                bin_pred = np.array(pred_probs)[mask]
                
                avg_true = np.mean(bin_true)
                avg_pred = np.mean(bin_pred)
                bin_error = np.abs(avg_true - avg_pred)
                
                # 加权误差
                weighted_error = (bin_size / len(true_probs)) * bin_error
                ece += weighted_error
                
                bin_details.append({
                    'bin': i,
                    'bin_range': (float(bin_edges[i]), float(bin_edges[i+1])),
                    'n_samples': int(bin_size),
                    'avg_true_prob': float(avg_true),
                    'avg_pred_prob': float(avg_pred),
                    'calibration_error': float(bin_error),
                    'weighted_error': float(weighted_error)
                })
        
        return {
            'ece': float(ece),
            'details': bin_details,
            'n_total_samples': int(len(true_probs))
        }
    
    def create_calibration_curve(self, true_probs: List[float], pred_probs: List[float]):
        """创建校准曲线数据"""
        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        bin_indices = np.digitize(pred_probs, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        
        prob_true = []
        prob_pred = []
        
        for i in range(self.n_bins):
            mask = bin_indices == i
            bin_size = np.sum(mask)
            
            if bin_size > 0:
                bin_true = np.array(true_probs)[mask]
                bin_pred = np.array(pred_probs)[mask]
                
                prob_true.append(float(np.mean(bin_true)))
                prob_pred.append(float(np.mean(bin_pred)))
            else:
                prob_true.append(np.nan)
                prob_pred.append(np.nan)
        
        # 移除空的分箱
        valid_mask = ~np.isnan(prob_true)
        prob_true = np.array(prob_true)[valid_mask]
        prob_pred = np.array(prob_pred)[valid_mask]
        
        return prob_true, prob_pred
    
    def load_and_align_data(self) -> Tuple[Dict, Dict]:
        """加载并对齐预测数据和真实数据"""
        config = config_manager.config
        
        print("Loading prediction data...")
        pred_file_path = config_manager.get_prediction_file_path()
        pred_df = pd.read_excel(pred_file_path, engine='openpyxl')
        
        print("Loading true survey data...")
        true_df = pd.read_excel(config.human_data_path, engine='openpyxl')
        
        print("Loading question mapping data...")
        question_df = pd.read_excel(config.question_data_path, sheet_name="Sheet1", engine="openpyxl")
        
        # 构建预测分布（使用qkey）
        pred_distributions = {}
        qkey_to_mapping = {}
        
        # 首先构建qkey到mapping的映射
        for _, row in question_df.iterrows():
            qkey = str(row["qkey"])
            mapping = self.safe_literal_eval(row['mapping'])
            qkey_to_mapping[qkey] = mapping
        
        # 构建预测分布
        for _, row in pred_df.iterrows():
            qkey = str(row['qkey'])
            dist = {}
            for col in row.index:
                if col.startswith('pred_'):
                    option = col.replace('pred_', '')
                    value = row[col]
                    if pd.isna(value):
                        value = 0.0
                    elif not isinstance(value, (int, float)):
                        try:
                            value = float(value)
                        except:
                            value = 0.0
                    dist[option] = max(0.0, min(1.0, float(value)))
            pred_distributions[qkey] = dist
        
        # 构建真实分布
        true_distributions = {}
        
        for _, row in true_df.iterrows():
            qkey = str(row['qkey'])
            if qkey in qkey_to_mapping:
                mapping = qkey_to_mapping[qkey]
                
                # 解析D_H分布
                d_h_str = row['D_H']
                d_h_probs = self.parse_distribution_string(d_h_str)
                
                # 构建真实分布（去掉refused选项）
                true_dist = {}
                valid_options = []
                
                # 识别并过滤掉refused选项
                for option_key, option_text in mapping.items():
                    option_text_lower = option_text.lower()
                    if not any(keyword.lower() in option_text_lower for keyword in self.refused_keywords):
                        valid_options.append(option_key)
                
                # 分配概率（假设D_H的顺序与valid_options的顺序一致）
                if len(d_h_probs) == len(valid_options):
                    for i, option in enumerate(valid_options):
                        true_dist[option] = float(d_h_probs[i])
                else:
                    # 如果长度不匹配，使用均匀分布
                    uniform_prob = 1.0 / len(valid_options) if valid_options else 1.0
                    for option in valid_options:
                        true_dist[option] = float(uniform_prob)
                
                true_distributions[qkey] = true_dist
        
        print(f"Predicted distributions: {len(pred_distributions)}")
        print(f"True distributions: {len(true_distributions)}")
        
        # 确保qkey一一对应
        common_keys = set(pred_distributions.keys()) & set(true_distributions.keys())
        print(f"Common keys: {len(common_keys)}")
        
        # 只保留共同的qkey
        pred_distributions = {k: v for k, v in pred_distributions.items() if k in common_keys}
        true_distributions = {k: v for k, v in true_distributions.items() if k in common_keys}
        
        return pred_distributions, true_distributions
    
    def collect_probability_pairs(self, pred_distributions: Dict, true_distributions: Dict) -> Tuple[List[float], List[float]]:
        """收集所有概率对"""
        all_true_probs = []
        all_pred_probs = []
        
        for qkey in pred_distributions.keys():
            if qkey in true_distributions:
                true_dist = true_distributions[qkey]
                pred_dist = pred_distributions[qkey]
                
                # 确保选项一一对应
                common_options = set(true_dist.keys()) & set(pred_dist.keys())
                
                for option in common_options:
                    all_true_probs.append(float(true_dist[option]))
                    all_pred_probs.append(float(pred_dist[option]))
        
        print(f"Total probability pairs: {len(all_true_probs)}")
        return all_true_probs, all_pred_probs
    
    def create_calibration_plot(self, true_probs: List[float], pred_probs: List[float]) -> Dict:
        """创建模型校准图"""
        config = config_manager.config
        
        if len(true_probs) < 2:
            print("Not enough data points to create calibration plot")
            return {}
        
        # 计算ECE
        ece_results = self.calculate_ece(true_probs, pred_probs)
        ece = ece_results['ece']
        
        # 创建校准曲线数据
        prob_true, prob_pred = self.create_calibration_curve(true_probs, pred_probs)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.plot_size)
        
        # === 图1: 散点图 + 校准曲线 ===
        # 绘制散点图（使用透明度避免重叠）
        ax1.scatter(pred_probs, true_probs, alpha=0.3, color='steelblue', s=20, label='Data points')
        
        # 绘制完美校准线
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated", linewidth=2)
        
        # 绘制校准曲线
        if len(prob_pred) > 0 and len(prob_true) > 0:
            ax1.plot(prob_pred, prob_true, "s-", linewidth=2, markersize=6, 
                    label=f"Model (ECE = {ece:.4f})")
        
        ax1.set_xlabel('Predicted Probability', fontsize=12)
        ax1.set_ylabel('True Probability', fontsize=12)
        ax1.set_title('Probability Calibration Scatter Plot', fontsize=14, fontweight='bold')
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # === 图2: 可靠性图 ===
        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_indices = np.digitize(pred_probs, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        
        bin_true_means = []
        bin_pred_means = []
        bin_counts = []
        
        for i in range(self.n_bins):
            mask = bin_indices == i
            bin_count = np.sum(mask)
            bin_counts.append(bin_count)
        
            if bin_count > 0:
                bin_true = np.array(true_probs)[mask]
                bin_pred = np.array(pred_probs)[mask]
                bin_true_means.append(float(np.mean(bin_true)))
                bin_pred_means.append(float(np.mean(bin_pred)))
            else:
                bin_true_means.append(np.nan)
                bin_pred_means.append(np.nan)
        
        # 移除空的分箱
        valid_mask = ~np.isnan(bin_true_means)
        bin_centers_valid = bin_centers[valid_mask]
        bin_true_means_valid = np.array(bin_true_means)[valid_mask]
        bin_counts_valid = np.array(bin_counts)[valid_mask]
        
        # 完美校准线
        ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration', alpha=0.8)
        
        # 普通折线图（无误差棒）
        ax2.plot(bin_centers_valid, bin_true_means_valid,
                 marker='o', linewidth=2, markersize=6,
                 label='Model Reliability', alpha=0.8)
        
        ax2.set_xlabel('Mean Predicted Probability (Binned)', fontsize=12)
        ax2.set_ylabel('Mean True Probability', fontsize=12)
        ax2.set_title('Binned Reliability Diagram', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        
        # 标注每个分箱的样本数
        for x, y, count in zip(bin_centers_valid, bin_true_means_valid, bin_counts_valid):
            ax2.annotate(f'n={int(count)}', (x, y),
                         textcoords="offset points",
                         xytext=(5, -15), ha='left', fontsize=8, alpha=0.7)

        plt.tight_layout()
        
        # 保存PNG文件
        os.makedirs(config.output_dir, exist_ok=True)
        plot_path = config_manager.get_plot_file_path()
        plt.savefig(plot_path, dpi=config.dpi, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"Model calibration plot saved to: {plot_path}")
        
        return ece_results
