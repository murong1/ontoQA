#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：验证Evaluation.py中的f1_score和normalize_answer函数逻辑
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from Evaluation import Evaluator
from collections import Counter


def test_normalize_answer():
    """测试normalize_answer函数"""
    print("=" * 60)
    print("测试normalize_answer函数")
    print("=" * 60)
    
    # 创建评估器实例
    evaluator = Evaluator("dummy_path", "test")
    
    # 测试用例
    test_cases = [
        ("CNBC", "CNBC"),
        ("CNBC Asia", "CNBC Asia"),
        ("Fifth largest continent in area.", "Fifth largest continent in area."),
        ("fifth-largest", "fifth-largest"),
        ("1982", "1982"),
        ("July 1, 1984", "July 1, 1984"),
        ("The quick brown fox", "The quick brown fox"),
        ("  Extra   spaces  ", "Extra spaces"),
        ("UPPERCASE text", "UPPERCASE text"),
        ("With-punctuation!", "With-punctuation!"),
        ("An article test", "An article test")
    ]
    
    for original, expected_desc in test_cases:
        normalized = evaluator.normalize_answer(original)
        print(f"原文: '{original}'")
        print(f"规范化后: '{normalized}'")
        print(f"描述: {expected_desc}")
        print("-" * 40)
    
    return evaluator


def test_f1_score_detailed():
    """详细测试f1_score函数的计算过程"""
    print("\n" + "=" * 60)
    print("详细测试f1_score函数的计算过程")
    print("=" * 60)
    
    evaluator = Evaluator("dummy_path", "test")
    
    # 测试用例
    test_cases = [
        ("CNBC", "CNBC Asia"),
        ("Fifth largest continent in area.", "fifth-largest"),
        ("1982", "July 1, 1984")
    ]
    
    for i, (prediction, ground_truth) in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}:")
        print(f"预测答案: '{prediction}'")
        print(f"标准答案: '{ground_truth}'")
        
        # 规范化处理
        norm_pred = evaluator.normalize_answer(prediction)
        norm_gt = evaluator.normalize_answer(ground_truth)
        print(f"规范化预测: '{norm_pred}'")
        print(f"规范化标准: '{norm_gt}'")
        
        # 分词
        pred_tokens = norm_pred.split()
        gt_tokens = norm_gt.split()
        print(f"预测词汇: {pred_tokens}")
        print(f"标准词汇: {gt_tokens}")
        
        # 计算交集
        pred_counter = Counter(pred_tokens)
        gt_counter = Counter(gt_tokens)
        common = pred_counter & gt_counter
        num_same = sum(common.values())
        
        print(f"预测词频: {dict(pred_counter)}")
        print(f"标准词频: {dict(gt_counter)}")
        print(f"共同词汇: {dict(common)}")
        print(f"共同词汇数量: {num_same}")
        
        # 计算指标
        if num_same == 0:
            f1, precision, recall = 0, 0, 0
            print("无共同词汇，所有指标为0")
        else:
            precision = num_same / len(pred_tokens)
            recall = num_same / len(gt_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            
            print(f"精确率计算: {num_same}/{len(pred_tokens)} = {precision:.4f}")
            print(f"召回率计算: {num_same}/{len(gt_tokens)} = {recall:.4f}")
            print(f"F1分数计算: (2 * {precision:.4f} * {recall:.4f}) / ({precision:.4f} + {recall:.4f}) = {f1:.4f}")
        
        # 使用函数验证结果
        f1_func, precision_func, recall_func = evaluator.f1_score(prediction, ground_truth)
        print(f"\n函数计算结果:")
        print(f"F1: {f1_func:.4f}, 精确率: {precision_func:.4f}, 召回率: {recall_func:.4f}")
        
        # 验证计算是否一致
        assert abs(f1 - f1_func) < 1e-6, f"F1计算不一致: {f1} vs {f1_func}"
        assert abs(precision - precision_func) < 1e-6, f"精确率计算不一致: {precision} vs {precision_func}"
        assert abs(recall - recall_func) < 1e-6, f"召回率计算不一致: {recall} vs {recall_func}"
        
        print("✓ 手工计算与函数结果一致")
        print("=" * 50)


def test_exact_match():
    """测试exact_match_score函数"""
    print("\n" + "=" * 60)
    print("测试exact_match_score函数")
    print("=" * 60)
    
    evaluator = Evaluator("dummy_path", "test")
    
    test_cases = [
        ("CNBC", "CNBC Asia"),
        ("Fifth largest continent in area.", "fifth-largest"),
        ("1982", "July 1, 1984"),
        ("CNBC", "cnbc"),  # 大小写
        ("The answer", "answer"),  # 冠词
        ("hello world!", "hello world")  # 标点符号
    ]
    
    for prediction, ground_truth in test_cases:
        em_score = evaluator.exact_match_score(prediction, ground_truth)
        norm_pred = evaluator.normalize_answer(prediction)
        norm_gt = evaluator.normalize_answer(ground_truth)
        
        print(f"预测: '{prediction}' -> 规范化: '{norm_pred}'")
        print(f"标准: '{ground_truth}' -> 规范化: '{norm_gt}'")
        print(f"完全匹配: {em_score}")
        print("-" * 40)


def test_eval_accuracy():
    """测试eval_accuracy函数（包含匹配）"""
    print("\n" + "=" * 60)
    print("测试eval_accuracy函数")
    print("=" * 60)
    
    evaluator = Evaluator("dummy_path", "test")
    
    test_cases = [
        ("CNBC is a news channel", "CNBC"),
        ("The answer is 1982", "1982"),
        ("Fifth largest continent", "fifth-largest"),
        ("No match here", "CNBC")
    ]
    
    for prediction, ground_truth in test_cases:
        accuracy = evaluator.eval_accuracy(prediction, ground_truth)
        norm_pred = evaluator.normalize_answer(prediction)
        norm_gt = evaluator.normalize_answer(ground_truth)
        
        print(f"预测: '{prediction}' -> 规范化: '{norm_pred}'")
        print(f"标准: '{ground_truth}' -> 规范化: '{norm_gt}'")
        print(f"准确率 (包含匹配): {accuracy}")
        print(f"'{norm_gt}' in '{norm_pred}': {norm_gt in norm_pred}")
        print("-" * 40)


def main():
    """主函数：运行所有测试"""
    print("开始测试Evaluation.py中的函数...")
    
    # 测试normalize_answer函数
    evaluator = test_normalize_answer()
    
    # 详细测试f1_score函数
    test_f1_score_detailed()
    
    # 测试exact_match_score函数
    test_exact_match()
    
    # 测试eval_accuracy函数
    test_eval_accuracy()
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()