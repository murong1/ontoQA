#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估模块
负责评估问答系统的性能
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import asyncio
from .Evaluation import Evaluator as EvaluationEvaluator


class Evaluator:
    """评估器"""
    
    def __init__(self, rag_system=None, mode="rag"):
        """
        初始化评估器
        
        Args:
            rag_system: RAG系统实例（zeroshot模式时可为None）
            mode: 评估模式，"rag"、"zeroshot"或"ablation"
        """
        self.rag_system = rag_system
        self.mode = mode
        self.logger = logging.getLogger(__name__)
        
        # 初始化zeroshot系统（如果需要）
        if mode == "zeroshot":
            from .zeroshot_qa import ZeroshotQA
            self.zeroshot_qa = ZeroshotQA()
    
    def evaluate(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        评估问答性能
        
        Args:
            questions: 问题列表
            
        Returns:
            Dict: 评估结果
        """
        self.logger.info(f"Starting evaluation on {len(questions)} questions")
        
        # 准备数据用于Evaluation.py
        evaluation_data = []
        
        # 根据模式使用不同的批处理方式
        if self.mode == "zeroshot":
            # Zero-shot模式使用批量处理
            evaluation_data = self._evaluate_zeroshot_batch(questions)
        elif self.mode in ["rag", "ablation"]:
            # RAG模式和消融实验模式都使用RAG批量处理
            evaluation_data = self._evaluate_rag_batch(questions)
        else:
            raise ValueError(f"Unsupported evaluation mode: {self.mode}")
        
        # 使用Evaluation.py进行详细评估
        detailed_metrics = self._run_detailed_evaluation(evaluation_data)
        
        # 获取配置信息
        from config import Config
        config_info = {
            'mode': self.mode,
            'question_llm_config': Config.get_question_llm_config(),
            'general_config': Config.get_config(),
            'experiment_description': getattr(Config, 'MIOASHU', 'No description')
        }
        
        # 在RAG和消融实验模式下添加embedding配置
        if self.mode in ["rag", "ablation"]:
            config_info['embedding_config'] = Config.get_embedding_config()
            
        # 仅在RAG模式下添加summary配置
        if self.mode == "rag":
            config_info['summary_llm_config'] = Config.get_summary_llm_config()
        
        evaluation_summary = {
            'total_questions': len(questions),
            'detailed_metrics': detailed_metrics,
            'timestamp': datetime.now().isoformat(),
            'config_info': config_info,
            'detailed_results': evaluation_data
        }
        
        # 添加主要指标到顶层以保持兼容性
        if 'accuracy' in detailed_metrics:
            evaluation_summary['accuracy'] = detailed_metrics['accuracy'] / 100  # 转换为0-1范围
            evaluation_summary['correct_answers'] = int(detailed_metrics['accuracy'] * len(questions) / 100)
        
        self.logger.info(f"Evaluation completed. Detailed metrics: {detailed_metrics}")
        
        return evaluation_summary
    
    def _evaluate_zeroshot_batch(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Zero-shot模式的批量评估
        
        Args:
            questions: 问题列表
            
        Returns:
            List[Dict[str, Any]]: 评估数据列表
        """
        self.logger.info(f"Running batch zeroshot evaluation on {len(questions)} questions")
        
        # 提取问题文本
        question_texts = [q.get('question', '') for q in questions]
        
        # 使用zeroshot_qa的批量处理方法
        self.logger.info("Calling ZeroshotQA batch processing...")
        qa_results = self.zeroshot_qa.answer_questions_batch(question_texts)
        
        # 构建评估数据
        evaluation_data = []
        for i, (question_data, qa_result) in enumerate(zip(questions, qa_results)):
            eval_item = {
                'question': question_data.get('question', ''),
                'answer': question_data.get('answer', ''),  # 标准答案
                'output': qa_result['answer'],  # 模型输出
                'question_id': question_data.get('id', i),
                'retrieved_documents': qa_result.get('retrieved_documents', []),
                'num_retrieved': qa_result.get('num_retrieved', 0),
                'mode': self.mode
            }
            evaluation_data.append(eval_item)
            
            # 每100个问题记录一次进度
            if (i + 1) % 100 == 0:
                self.logger.info(f"Processed {i+1}/{len(questions)} questions")
        
        self.logger.info(f"Completed batch processing of {len(questions)} questions")
        return evaluation_data
    
    def _evaluate_rag_batch(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        RAG模式的批量评估
        
        Args:
            questions: 问题列表
            
        Returns:
            List[Dict[str, Any]]: 评估数据列表
        """
        self.logger.info(f"Running batch RAG evaluation on {len(questions)} questions")
        
        # 提取问题文本
        question_texts = [q.get('question', '') for q in questions]
        
        # 使用rag_system的批量处理方法
        self.logger.info("Calling RAG system batch processing...")
        qa_results = self.rag_system.answer_questions_batch(question_texts)
        
        # 构建评估数据
        evaluation_data = []
        for i, (question_data, qa_result) in enumerate(zip(questions, qa_results)):
            eval_item = {
                'question': question_data.get('question', ''),
                'answer': question_data.get('answer', ''),  # 标准答案
                'output': qa_result['answer'],  # 模型输出
                'question_id': question_data.get('id', i),
                'retrieved_documents': qa_result.get('retrieved_documents', []),
                'num_retrieved': qa_result.get('num_retrieved', 0),
                'mode': self.mode
            }
            evaluation_data.append(eval_item)
            
            # 每100个问题记录一次进度
            if (i + 1) % 100 == 0:
                self.logger.info(f"Processed {i+1}/{len(questions)} questions")
        
        self.logger.info(f"Completed batch processing of {len(questions)} questions")
        return evaluation_data
    
    def _run_detailed_evaluation(self, evaluation_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        使用Evaluation.py运行详细评估
        
        Args:
            evaluation_data: 评估数据列表
            
        Returns:
            Dict[str, float]: 详细评估指标
        """
        try:
            # 创建临时JSON Lines文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_file = Path(f"temp_eval_{timestamp}.jsonl")
            
            # 写入数据到临时文件
            with open(temp_file, 'w', encoding='utf-8') as f:
                for item in evaluation_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            # 创建Evaluation.py的评估器
            eval_evaluator = EvaluationEvaluator(str(temp_file), "musique")
            
            # 运行评估（需要异步运行）
            try:
                # 尝试同步运行评估
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果已有事件循环，使用同步方式
                    import pandas as pd
                    df = pd.read_json(temp_file, lines=True)
                    result_dict, _ = eval_evaluator.short_eval(df)
                else:
                    # 如果没有事件循环，创建新的
                    result_dict = asyncio.run(eval_evaluator.evaluate())
            except Exception as e:
                # 如果异步运行失败，使用同步评估
                self.logger.warning(f"Async evaluation failed, using sync: {e}")
                import pandas as pd
                df = pd.read_json(temp_file, lines=True)
                result_dict, _ = eval_evaluator.short_eval(df)
            
            # 清理临时文件
            if temp_file.exists():
                temp_file.unlink()
            
            return result_dict
            
        except Exception as e:
            self.logger.error(f"Detailed evaluation failed: {e}")
            # 返回简化评估结果
            return self._simple_evaluation_fallback(evaluation_data)
    
    def _simple_evaluation_fallback(self, evaluation_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        简化评估的备用方案
        
        Args:
            evaluation_data: 评估数据列表
            
        Returns:
            Dict[str, float]: 简化评估指标
        """
        correct = 0
        total = len(evaluation_data)
        
        for item in evaluation_data:
            if self._evaluate_answer(item['output'], item['answer']):
                correct += 1
        
        accuracy = (correct / total * 100) if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'em': 0.0
        }
    
    def _evaluate_answer(self, generated_answer: str, expected_answer: str) -> bool:
        """
        评估生成的答案是否正确
        
        Args:
            generated_answer: 生成的答案
            expected_answer: 期望的答案
            
        Returns:
            bool: 是否正确
        """
        # 简化的评估逻辑：检查期望答案是否在生成答案中
        if not expected_answer or not generated_answer:
            return False
        
        return expected_answer.lower() in generated_answer.lower()
    
    def save_results(self, results: Dict[str, Any], output_dir: Path):
        """
        保存评估结果
        
        Args:
            results: 评估结果
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 提取数据集名称
        dataset_name = self._extract_dataset_name_from_config(results.get('config_info', {}))
        
        # 保存详细结果
        results_file = output_dir / f'evaluation_results_{dataset_name}_{self.mode}_{timestamp}.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存摘要报告
        summary_file = output_dir / f'evaluation_summary_{dataset_name}_{self.mode}_{timestamp}.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            mode_map = {
                "zeroshot": "Zero-shot Baseline",
                "rag": "OntoQA RAG System",
                "ablation": "Ablation Study (Documents Only)"
            }
            mode_name = mode_map.get(results['config_info']['mode'], "Unknown Mode")
            f.write(f"{mode_name} Evaluation Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Mode: {results['config_info']['mode'].upper()}\n")
            f.write(f"Experiment Description: {results['config_info']['experiment_description']}\n\n")
            
            # 结果摘要
            f.write("Results:\n")
            f.write("-" * 10 + "\n")
            f.write(f"Total Questions: {results['total_questions']}\n")
            
            # 显示详细评估指标
            if 'detailed_metrics' in results:
                detailed = results['detailed_metrics']
                f.write(f"Accuracy: {detailed.get('accuracy', 0):.2f}%\n")
                f.write(f"F1 Score: {detailed.get('f1', 0):.2f}%\n")
                f.write(f"Precision: {detailed.get('precision', 0):.2f}%\n")
                f.write(f"Recall: {detailed.get('recall', 0):.2f}%\n")
                f.write(f"Exact Match: {detailed.get('em', 0):.2f}%\n")
            else:
                # 兼容旧格式
                f.write(f"Correct Answers: {results.get('correct_answers', 0)}\n")
                f.write(f"Accuracy: {results.get('accuracy', 0):.3f}\n")
            f.write("\n")
            
            # 配置信息
            f.write("Configuration:\n")
            f.write("-" * 15 + "\n")
            
            # LLM配置
            question_config = results['config_info']['question_llm_config']
            f.write(f"Question LLM: {question_config['model']}\n")
            f.write(f"Question API: {question_config['api_url']}\n")
            f.write(f"Question Max Tokens: {question_config['max_tokens']}\n\n")
            
            # 根据模式显示不同配置
            if results['config_info']['mode'] in ["rag", "ablation"]:
                # Embedding配置
                if 'embedding_config' in results['config_info']:
                    embed_config = results['config_info']['embedding_config']
                    f.write(f"Embedding Model: {embed_config['model_name']}\n")
                    f.write(f"Embedding API: {embed_config['api_url']}\n")
                    f.write(f"Embedding Batch Size: {embed_config['batch_size']}\n\n")
                
                # Summary LLM配置（仅在RAG模式下显示）
                if results['config_info']['mode'] == "rag" and 'summary_llm_config' in results['config_info']:
                    summary_config = results['config_info']['summary_llm_config']
                    f.write(f"Summary LLM: {summary_config['model']}\n")
                    f.write(f"Summary Max Tokens: {summary_config['max_tokens']}\n\n")
                
                # 通用配置
                general_config = results['config_info']['general_config']
                if results['config_info']['mode'] == "rag":
                    f.write(f"Number of Clusters: {general_config['n_clusters']}\n")
                f.write(f"Top-K Retrieval: {general_config['top_k']}\n")
                f.write(f"Corpus Path: {general_config['corpus_path']}\n")
            
            # 通用配置
            general_config = results['config_info']['general_config']
            f.write(f"Questions Path: {general_config['questions_path']}\n")
        
        self.logger.info(f"Results saved to {output_dir}")
        
        return results_file, summary_file
    
    def _extract_dataset_name_from_config(self, config_info: Dict[str, Any]) -> str:
        """
        从配置信息中提取数据集名称
        
        Args:
            config_info: 配置信息字典
            
        Returns:
            str: 数据集名称
        """
        try:
            # 从配置中获取语料库路径
            general_config = config_info.get('general_config', {})
            if self.mode == 'zeroshot':
                # zeroshot模式下从 questions_path 提取
                questions_path = general_config.get('questions_path', '')
                if questions_path:
                    return self._extract_dataset_name_from_path(questions_path)
            else:
                # RAG模式下从 corpus_path 提取
                corpus_path = general_config.get('corpus_path', '')
                if corpus_path:
                    return self._extract_dataset_name_from_path(corpus_path)
        except Exception as e:
            self.logger.warning(f"Failed to extract dataset name from config: {e}")
        
        return 'unknown'
    
    def _extract_dataset_name_from_path(self, file_path: str) -> str:
        """
        从文件路径中提取数据集名称
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: 数据集名称
        """
        path_obj = Path(file_path)
        
        # 检查父目录是否是已知的数据集名称
        parent_name = path_obj.parent.name.lower()
        known_datasets = ['musique', 'quality', 'hotpot', 'nq', 'triviaqa']
        
        if parent_name in known_datasets:
            return parent_name
        
        # 如果父目录不是已知数据集，尝试从文件名中提取
        file_stem = path_obj.stem.lower()
        for dataset in known_datasets:
            if dataset in file_stem:
                return dataset
        
        # 如果都无法识别，返回文件名的简化版本
        return file_stem.split('_')[0] if '_' in file_stem else file_stem