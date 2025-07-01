#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本体生成主程序
仅进行文档聚类和本体总结，不包含问答操作
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime

from src.data_loader import DataLoader
from src.clustering import TextClusterer
from src.summarizer import OntologySummarizer
from src.owl_converter import JSONToOWLConverter
from config import Config


def setup_logging(level=logging.INFO):
    """设置日志配置"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('ontology_generation.log', encoding='utf-8')
        ]
    )


def main():
    """主函数 - 仅生成本体"""
    parser = argparse.ArgumentParser(description='OntoQA: 本体生成器')
    parser.add_argument('--corpus', type=str, 
                       default=Config.DEFAULT_CORPUS_PATH,
                       help='语料库文件路径')
    parser.add_argument('--clusters', type=int, default=Config.DEFAULT_N_CLUSTERS,
                       help='聚类数量 (默认根据文档数量自动调整为1/5到1/10)')
    parser.add_argument('--max-cluster-size', type=int, default=20,
                       help='单个聚类的最大文档数量 (默认: 20)')
    parser.add_argument('--min-cluster-size', type=int, default=None,
                       help='单个聚类的最小文档数量 (默认: 自动计算)')
    parser.add_argument('--balanced-clustering', action='store_true', default=True,
                       help='启用平衡聚类算法 (默认: 启用)')
    parser.add_argument('--use-gpu', action='store_true',
                       help='启用GPU加速聚类 (需要安装cuml)')
    parser.add_argument('--clustering-algorithm', type=str, default='auto',
                       choices=['auto', 'kmeans', 'minibatch', 'faiss'],
                       help='聚类算法选择 (默认: auto)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='CPU并行任务数，-1使用所有核心 (默认: -1)')
    parser.add_argument('--output', type=str, default=Config.DEFAULT_OUTPUT_DIR,
                       help='输出目录')
    parser.add_argument('--verbose', action='store_true',
                       help='启用详细日志')
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("[本体生成] 开始本体生成任务...")
        
        # 1. 数据加载
        logger.info("[本体生成] 加载语料库数据...")
        data_loader = DataLoader()
        corpus = data_loader.load_corpus(args.corpus)
        logger.info(f"[本体生成] 加载了 {len(corpus)} 个文档")
        
        # 2. 根据文档数量和最大聚类大小调整聚类数量
        if args.clusters == Config.DEFAULT_N_CLUSTERS:  # 如果使用默认值，则动态调整
            # 基于最大聚类大小计算最小聚类数量
            min_clusters_by_size = (len(corpus) + args.max_cluster_size - 1) // args.max_cluster_size
            min_clusters = max(3, min_clusters_by_size, len(corpus) // 10)  # 最少3个聚类
            max_clusters = max(5, len(corpus) // 5)   # 最多文档数的1/5
            args.clusters = min(max_clusters, max(min_clusters, Config.DEFAULT_N_CLUSTERS))
            logger.info(f"[本体生成] 根据 {len(corpus)} 个文档和最大聚类大小 {args.max_cluster_size} 自动调整聚类数为 {args.clusters}")
        
        # 3. 文本聚类（使用平衡聚类）
        if args.balanced_clustering:
            logger.info(f"[本体生成] 使用平衡聚类算法将文档聚类为 {args.clusters} 个聚类...")
            logger.info(f"[本体生成] 聚类大小限制: 最小={args.min_cluster_size or '自动'}, 最大={args.max_cluster_size}")
        else:
            logger.info(f"[本体生成] 使用标准聚类算法将文档聚类为 {args.clusters} 个聚类...")
        
        clusterer = TextClusterer(
            n_clusters=args.clusters, 
            output_dir=args.output,
            min_cluster_size=args.min_cluster_size,
            max_cluster_size=args.max_cluster_size,
            balanced_clustering=args.balanced_clustering,
            use_gpu=args.use_gpu,
            n_jobs=args.n_jobs,
            algorithm=args.clustering_algorithm
        )
        clusters = clusterer.cluster_documents(corpus)
        
        # 打印聚类统计信息
        cluster_stats = clusterer.get_clustering_stats(clusters)
        logger.info(f"[本体生成] 聚类完成 - 聚类大小分布: {cluster_stats['cluster_sizes']}")
        logger.info(f"[本体生成] 最小聚类: {cluster_stats['min_cluster_size']}, 最大聚类: {cluster_stats['max_cluster_size']}, 平均: {cluster_stats['avg_cluster_size']:.1f}")
        
        # 4. 本体总结
        logger.info("[本体生成] 为聚类生成本体总结...")
        summarizer = OntologySummarizer(output_dir=args.output)
        summaries = summarizer.summarize_clusters(clusters)
        logger.info("[本体生成] 本体总结完成")
        
        # 5. 保存本体总结结果为JSON格式
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        
        # 构建详细的本体JSON数据
        ontology_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "corpus_file": args.corpus,
                "document_count": len(corpus),
                "cluster_count": args.clusters,
                "ontology_count": len(summaries)
            },
            "ontologies": []
        }
        
        # 提取每个本体的详细信息
        for cluster_id, summary_data in summaries.items():
            # 转换所有数值为Python原生类型，避免numpy类型序列化问题
            ontology_info = {
                "cluster_id": int(cluster_id),
                "ontology_name": str(summary_data.get('ontology_name', f'Cluster_{cluster_id}')),
                "description": str(summary_data.get('ontology_summary', '')),
                "key_concepts": [str(concept) for concept in summary_data.get('key_concepts', [])],
                "relationships": [str(rel) for rel in summary_data.get('relationships', [])],
                "document_count": int(summary_data.get('document_count', 0)),
                "source_clusters": [int(sc) for sc in summary_data.get('source_clusters', [cluster_id])]
            }
            ontology_data["ontologies"].append(ontology_info)
        
        # 保存JSON文件
        ontology_json_file = output_dir / "ontology_summaries.json"
        with open(ontology_json_file, 'w', encoding='utf-8') as f:
            json.dump(ontology_data, f, ensure_ascii=False, indent=2)
        
        # 同时保存简化的文本版本用于快速查看
        ontology_txt_file = output_dir / "ontology_summaries.txt"
        with open(ontology_txt_file, 'w', encoding='utf-8') as f:
            f.write("=== 本体总结结果 ===\n\n")
            for ontology in ontology_data["ontologies"]:
                f.write(f"本体名称: {ontology['ontology_name']}\n")
                f.write(f"描述: {ontology['description']}\n")
                if ontology['key_concepts']:
                    f.write(f"关键概念: {', '.join(ontology['key_concepts'])}\n")
                if ontology['relationships']:
                    f.write("关系:\n")
                    for rel in ontology['relationships']:
                        f.write(f"  - {rel}\n")
                f.write(f"文档数量: {ontology['document_count']}\n")
                f.write("-" * 50 + "\n\n")
        
        logger.info(f"[本体生成] 本体生成完成。共生成 {len(summaries)} 个本体总结")
        logger.info(f"[本体生成] JSON结果已保存到 {ontology_json_file}")
        logger.info(f"[本体生成] 文本结果已保存到 {ontology_txt_file}")
        
        # 6. OWL转换
        logger.info("[本体生成] 开始将JSON本体转换为OWL格式...")
        owl_output_dir = output_dir / "onto"
        owl_output_dir.mkdir(exist_ok=True)
        
        try:
            owl_converter = JSONToOWLConverter()
            owl_files = owl_converter.convert_json_to_owl(
                json_file_path=str(ontology_json_file), 
                output_dir=str(owl_output_dir)
            )
            logger.info(f"[本体生成] OWL转换完成")
            logger.info(f"[本体生成] Turtle文件: {owl_files['turtle']}")
            logger.info(f"[本体生成] OWL/XML文件: {owl_files['owl_xml']}")
        except Exception as e:
            logger.error(f"[本体生成] OWL转换失败: {str(e)}")
            owl_files = None
        
        # 7. 打印简要统计信息
        print(f"\n=== 本体生成统计 ===")
        print(f"文档数量: {len(corpus)}")
        print(f"聚类数量: {args.clusters}")
        print(f"聚类算法: {'平衡聚类' if args.balanced_clustering else '标准KMeans'}")
        if args.balanced_clustering:
            print(f"聚类大小限制: {args.min_cluster_size or '自动'} - {args.max_cluster_size}")
            print(f"实际聚类大小: {cluster_stats['min_cluster_size']} - {cluster_stats['max_cluster_size']} (平均: {cluster_stats['avg_cluster_size']:.1f})")
        print(f"本体总结数量: {len(summaries)}")
        print(f"输出目录: {args.output}")
        print(f"JSON文件: {ontology_json_file}")
        print(f"文本文件: {ontology_txt_file}")
        if owl_files:
            print(f"OWL Turtle文件: {owl_files['turtle']}")
            print(f"OWL XML文件: {owl_files['owl_xml']}")
        
    except Exception as e:
        logger.error(f"[本体生成] 任务失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()