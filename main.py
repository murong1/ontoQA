#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主程序入口文件
通过聚类相似文本片段，使用LLM总结聚类的本体，最终进行RAG操作来增强模型能力
"""

import argparse
import logging
import sys
from pathlib import Path

from src.data_loader import DataLoader
from src.clustering import TextClusterer
from src.summarizer import OntologySummarizer
from src.rag_system import RAGSystem
from src.evaluator import Evaluator
from config import Config


def setup_logging(level=logging.INFO):
    """设置日志配置"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('experiment.log', encoding='utf-8')
        ]
    )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='OntoQA: Ontology-enhanced RAG System')
    parser.add_argument('--corpus', type=str, 
                       default=Config.DEFAULT_CORPUS_PATH,
                       help='Path to corpus file')
    parser.add_argument('--questions', type=str,
                       default=Config.DEFAULT_QUESTIONS_PATH, 
                       help='Path to questions file')
    parser.add_argument('--clusters', type=int, default=Config.DEFAULT_N_CLUSTERS,
                       help='Number of clusters (auto-adjusted to 1/5 to 1/10 of document count if using default)')
    parser.add_argument('--output', type=str, default=Config.DEFAULT_OUTPUT_DIR,
                       help='Output directory for results')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--zeroshot', action='store_true',
                       help='Enable zero-shot mode (direct QA without retrieval for baseline)')
    parser.add_argument('--ablation', action='store_true',
                       help='Enable ablation study mode (RAG with documents only, no ontology summaries)')
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    try:
        if args.zeroshot:
            logger.info("[主程序] 开始 OntoQA Zero-shot 基线实验...")
            
            # 1. 数据加载（仅加载问题）
            logger.info("[主程序] 加载问题数据...")
            data_loader = DataLoader()
            questions = data_loader.load_questions(args.questions)
            logger.info(f"[主程序] 加载了 {len(questions)} 个问题")
            
            # 2. Zero-shot问答评估
            logger.info("[主程序] 执行 zero-shot 问答评估...")
            evaluator = Evaluator(None, mode="zeroshot")  # 不传入RAG系统
            results = evaluator.evaluate(questions)
            
            # 3. 保存结果
            output_dir = Path(args.output)
            output_dir.mkdir(exist_ok=True)
            evaluator.save_results(results, output_dir)
            
            logger.info(f"[主程序] Zero-shot 基线实验完成。结果已保存到 {output_dir}")
        
        elif args.ablation:
            logger.info("[主程序] 开始 OntoQA 消融实验（仅文档）...")
            
            # 数据加载
            logger.info("[主程序] 加载数据...")
            data_loader = DataLoader()
            corpus = data_loader.load_corpus(args.corpus)
            questions = data_loader.load_questions(args.questions)
            logger.info(f"[主程序] 加载了 {len(corpus)} 个文档和 {len(questions)} 个问题")
            
            # 构建RAG索引（仅包含原始文档）
            logger.info("[主程序] 构建RAG系统（仅文档，无本体总结）...")
            rag_system = RAGSystem(top_k=Config.DEFAULT_TOP_K, output_dir=args.output)
            rag_system.build_index_documents_only(corpus, args.corpus)
            logger.info("[主程序] RAG系统就绪")
            
            # 问答评估
            logger.info("[主程序] 执行问答评估...")
            evaluator = Evaluator(rag_system, mode="ablation")
            results = evaluator.evaluate(questions)
            
            # 保存结果
            output_dir = Path(args.output)
            output_dir.mkdir(exist_ok=True)
            evaluator.save_results(results, output_dir)
            
            logger.info(f"[主程序] 消融实验完成。结果已保存到 {output_dir}")
        
        else:
            logger.info("[主程序] 开始 OntoQA 完整实验...")
            
            # 数据加载
            logger.info("[主程序] 加载数据...")
            data_loader = DataLoader()
            corpus = data_loader.load_corpus(args.corpus)
            questions = data_loader.load_questions(args.questions)
            logger.info(f"[主程序] 加载了 {len(corpus)} 个文档和 {len(questions)} 个问题")
            
            # 根据文档数量调整聚类数量（文档数的1/5到1/10）
            if args.clusters == Config.DEFAULT_N_CLUSTERS:  # 如果使用默认值，则动态调整
                min_clusters = max(3, len(corpus) // 10)  # 最少3个聚类
                max_clusters = max(5, len(corpus) // 5)   # 最多文档数的1/5
                args.clusters = min(max_clusters, max(min_clusters, Config.DEFAULT_N_CLUSTERS))
                logger.info(f"[主程序] 根据 {len(corpus)} 个文档自动调整聚类数为 {args.clusters}")
            
            # 文本聚类
            logger.info(f"[主程序] 将文档聚类为 {args.clusters} 个聚类...")
            clusterer = TextClusterer(n_clusters=args.clusters, output_dir=args.output)
            clusters = clusterer.cluster_documents(corpus)
            logger.info("[主程序] 聚类完成")
            
            # 本体总结
            logger.info("[主程序] 为聚类生成本体总结...")
            summarizer = OntologySummarizer(output_dir=args.output)
            summaries = summarizer.summarize_clusters(clusters)
            logger.info("[主程序] 本体总结完成")
            
            # 构建RAG索引（自动检查缓存）
            logger.info("[主程序] 构建带FAISS索引的RAG系统...")
            rag_system = RAGSystem(top_k=Config.DEFAULT_TOP_K, output_dir=args.output)
            rag_system.build_index(corpus, summaries, args.corpus, args.clusters)
            logger.info("[主程序] RAG系统就绪")
            
            # 问答评估
            logger.info("[主程序] 执行问答评估...")
            evaluator = Evaluator(rag_system)
            results = evaluator.evaluate(questions)
            
            # 保存结果
            output_dir = Path(args.output)
            output_dir.mkdir(exist_ok=True)
            evaluator.save_results(results, output_dir)
            
            logger.info(f"[主程序] 实验完成。结果已保存到 {output_dir}")
        
    except Exception as e:
        logger.error(f"[主程序] 实验失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()