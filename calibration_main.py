# calibration_main.py
import os
from config.calibration_config import config_manager
from core.calibration_analyzer import ECEAnalyzer

def main():
    """主函数 - 模型校准分析"""
    config = config_manager.config
    
    print("=== 模型校准分析系统 ===")
    print(f"模型: {config_manager.get_model_display_name()}")
    print(f"输出类型: {config.output_type}")
    print(f"分箱数: {config.n_bins}")
    
    # 检查预测文件是否存在
    pred_file_path = config_manager.get_prediction_file_path()
    if not os.path.exists(pred_file_path):
        print(f"错误：预测文件不存在: {pred_file_path}")
        print("请先运行预测脚本生成预测数据")
        return
    
    # 检查真实数据文件是否存在
    if not os.path.exists(config.human_data_path):
        print(f"错误：真实数据文件不存在: {config.human_data_path}")
        return
    
    # 创建校准分析器
    analyzer = ECEAnalyzer(n_bins=config.n_bins)
    
    # 加载数据
    print("\n正在加载数据...")
    pred_distributions, true_distributions = analyzer.load_and_align_data()
    
    if len(pred_distributions) == 0 or len(true_distributions) == 0:
        print("错误：没有找到匹配的数据进行校准分析")
        return
    
    # 收集概率对
    all_true_probs, all_pred_probs = analyzer.collect_probability_pairs(
        pred_distributions, true_distributions
    )
    
    if len(all_true_probs) == 0:
        print("错误：没有有效的概率对进行分析")
        return
    
    # 生成校准图
    print("\n正在生成校准图...")
    ece_results = analyzer.create_calibration_plot(all_true_probs, all_pred_probs)
    
    # 输出统计信息
    print(f"\n=== 校准分析结果 ===")
    print(f"模型: {config_manager.get_model_display_name()}")
    print(f"输出类型: {config.output_type}")
    print(f"总数据点: {len(all_true_probs)}")
    print(f"分析的问题数: {len(pred_distributions)}")
    print(f"期望校准误差 (ECE): {ece_results['ece']:.4f}")
    
    # 输出分箱详情
    print(f"\n分箱详情 (前5个):")
    for detail in ece_results['details'][:5]:
        bin_range = detail['bin_range']
        print(f"  分箱 {detail['bin']}: [{bin_range[0]:.1f}-{bin_range[1]:.1f}] | "
              f"样本数: {detail['n_samples']} | "
              f"真实概率: {detail['avg_true_prob']:.3f} | "
              f"预测概率: {detail['avg_pred_prob']:.3f} | "
              f"校准误差: {detail['calibration_error']:.3f}")
    
    if len(ece_results['details']) > 5:
        print(f"  ... 还有 {len(ece_results['details']) - 5} 个分箱")
    
    print(f"\n校准图已保存至: {config_manager.get_plot_file_path()}")

if __name__ == "__main__":
    # 在这里修改配置即可切换不同模型和输出类型
    # config_manager.update_config(model_type="deepseek", output_type="logits")  # Deepseek API + logits
    # config_manager.update_config(model_type="qwen7b", output_type="probability")  # Qwen 7B + 概率
    config_manager.update_config(model_type="qwen32b", output_type="logits", use_multi_gpu=True)  # Qwen 32B + logits + 多GPU
    
    main()
