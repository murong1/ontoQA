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
            logger.info("Starting OntoQA Zero-shot Baseline experiment...")
            
            # 1. 数据加载（仅加载问题）
            logger.info("Loading questions...")
            data_loader = DataLoader()
            questions = data_loader.load_questions(args.questions)
            logger.info(f"Loaded {len(questions)} questions")
            
            # 2. Zero-shot问答评估
            logger.info("Running zero-shot question answering evaluation...")
            evaluator = Evaluator(None, mode="zeroshot")  # 不传入RAG系统
            results = evaluator.evaluate(questions)
            
            # 3. 保存结果
            output_dir = Path(args.output)
            output_dir.mkdir(exist_ok=True)
            evaluator.save_results(results, output_dir)
            
            logger.info(f"Zero-shot baseline experiment completed. Results saved to {output_dir}")
        
        elif args.ablation:
            logger.info("Starting OntoQA Ablation Study experiment (documents only)...")
            
            # 数据加载
            logger.info("Loading data...")
            data_loader = DataLoader()
            corpus = data_loader.load_corpus(args.corpus)
            questions = data_loader.load_questions(args.questions)
            logger.info(f"Loaded {len(corpus)} documents and {len(questions)} questions")
            
            # 构建RAG索引（仅包含原始文档）
            logger.info("Building RAG system with documents only (no ontology summaries)...")
            rag_system = RAGSystem()
            rag_system.build_index_documents_only(corpus, args.corpus)
            logger.info("RAG system ready")
            
            # 问答评估
            logger.info("Running question answering evaluation...")
            evaluator = Evaluator(rag_system, mode="ablation")
            results = evaluator.evaluate(questions)
            
            # 保存结果
            output_dir = Path(args.output)
            output_dir.mkdir(exist_ok=True)
            evaluator.save_results(results, output_dir)
            
            logger.info(f"Ablation study experiment completed. Results saved to {output_dir}")
        
        else:
            logger.info("Starting OntoQA experiment...")
            
            # 数据加载
            logger.info("Loading data...")
            data_loader = DataLoader()
            corpus = data_loader.load_corpus(args.corpus)
            questions = data_loader.load_questions(args.questions)
            logger.info(f"Loaded {len(corpus)} documents and {len(questions)} questions")
            
            # 根据文档数量调整聚类数量（文档数的1/5到1/10）
            if args.clusters == Config.DEFAULT_N_CLUSTERS:  # 如果使用默认值，则动态调整
                min_clusters = max(3, len(corpus) // 10)  # 最少3个聚类
                max_clusters = max(5, len(corpus) // 5)   # 最多文档数的1/5
                args.clusters = min(max_clusters, max(min_clusters, Config.DEFAULT_N_CLUSTERS))
                logger.info(f"Auto-adjusted clusters to {args.clusters} based on {len(corpus)} documents")
            
            # 文本聚类
            logger.info(f"Clustering documents into {args.clusters} clusters...")
            clusterer = TextClusterer(n_clusters=args.clusters)
            clusters = clusterer.cluster_documents(corpus)
            logger.info("Clustering completed")
            
            # 本体总结
            logger.info("Generating ontology summaries for clusters...")
            summarizer = OntologySummarizer()
            summaries = summarizer.summarize_clusters(clusters)
            logger.info("Ontology summarization completed")
            
            # 构建RAG索引（自动检查缓存）
            logger.info("Building RAG system with FAISS index...")
            rag_system = RAGSystem()
            rag_system.build_index(corpus, summaries, args.corpus, args.clusters)
            logger.info("RAG system ready")
            
            # 问答评估
            logger.info("Running question answering evaluation...")
            evaluator = Evaluator(rag_system)
            results = evaluator.evaluate(questions)
            
            # 保存结果
            output_dir = Path(args.output)
            output_dir.mkdir(exist_ok=True)
            evaluator.save_results(results, output_dir)
            
            logger.info(f"Experiment completed. Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()