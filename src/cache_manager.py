#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存管理模块
"""

import os
import json
import hashlib
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from config import Config


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, output_dir: str = "results"):
        """
        初始化缓存管理器
        
        Args:
            output_dir: 输出根目录
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        self.cache_dir = os.path.join(output_dir, Config.CACHE_DIR, Config.ONTOLOGY_CACHE_DIR)
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def generate_cluster_signature(self, clusters: Dict[int, List[Dict[str, Any]]]) -> str:
        """
        为聚类数据生成唯一签名
        
        Args:
            clusters: 聚类数据
            
        Returns:
            str: 唯一签名字符串
        """
        cluster_content = []
        for cluster_id in sorted(clusters.keys()):
            docs = clusters[cluster_id]
            doc_contents = []
            for doc in docs:
                content_str = f"{doc.get('title', '')}{doc.get('context', '')}"
                doc_contents.append(content_str)
            cluster_content.append(f"cluster_{cluster_id}:" + "|".join(sorted(doc_contents)))
        
        signature_str = "||".join(cluster_content)
        signature_hash = hashlib.md5(signature_str.encode('utf-8')).hexdigest()[:12]
        
        return signature_hash
    
    def load_extraction_cache(self, signature: str) -> Optional[List[Dict[str, Any]]]:
        """
        加载第一阶段本体抽取缓存
        
        Args:
            signature: 数据签名
            
        Returns:
            Optional[List[Dict]]: 缓存的本体列表，如果不存在则返回None
        """
        cache_file = os.path.join(self.cache_dir, f"stage1_raw_ontologies_{signature}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                self.logger.info(f"[缓存] 从缓存加载第一阶段结果: {cache_file}")
                return cached_data
            except Exception as e:
                self.logger.warning(f"[缓存] 加载第一阶段缓存失败: {e}")
        
        return None
    
    def save_extraction_cache(self, signature: str, raw_ontologies: List[Dict[str, Any]]) -> None:
        """
        保存第一阶段本体抽取缓存
        
        Args:
            signature: 数据签名
            raw_ontologies: 原始本体列表
        """
        cache_file = os.path.join(self.cache_dir, f"stage1_raw_ontologies_{signature}.json")
        
        try:
            cleaned_ontologies = self._clean_data_for_json(raw_ontologies)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned_ontologies, f, ensure_ascii=False, indent=2)
            self.logger.info(f"[缓存] 第一阶段结果已缓存: {cache_file}")
        except Exception as e:
            self.logger.error(f"[缓存] 保存第一阶段缓存失败: {e}")
    
    def load_merge_cache(self, signature: str) -> Optional[List[Dict[str, Any]]]:
        """
        加载第二阶段本体合并缓存
        
        Args:
            signature: 数据签名
            
        Returns:
            Optional[List[Dict]]: 缓存的合并本体列表，如果不存在则返回None
        """
        cache_file = os.path.join(self.cache_dir, f"stage2_merged_ontologies_{signature}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                self.logger.info(f"[缓存] 从缓存加载第二阶段结果: {cache_file}")
                return cached_data
            except Exception as e:
                self.logger.warning(f"[缓存] 加载第二阶段缓存失败: {e}")
        
        return None
    
    def save_merge_cache(self, signature: str, merged_ontologies: List[Dict[str, Any]]) -> None:
        """
        保存第二阶段本体合并缓存
        
        Args:
            signature: 数据签名
            merged_ontologies: 合并后的本体列表
        """
        cache_file = os.path.join(self.cache_dir, f"stage2_merged_ontologies_{signature}.json")
        
        try:
            cleaned_ontologies = self._clean_data_for_json(merged_ontologies)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned_ontologies, f, ensure_ascii=False, indent=2)
            self.logger.info(f"[缓存] 第二阶段结果已缓存: {cache_file}")
        except Exception as e:
            self.logger.error(f"[缓存] 保存第二阶段缓存失败: {e}")
    
    def save_processing_details(self, original_clusters: Dict[int, List[Dict[str, Any]]], 
                               raw_ontologies: List[Dict[str, Any]], 
                               merged_ontologies: List[Dict[str, Any]], run_dir: str = None) -> None:
        """
        保存本体处理详情
        
        Args:
            original_clusters: 原始聚类结果
            raw_ontologies: 原始本体列表
            merged_ontologies: 合并后的本体列表
            run_dir: 运行目录，优先使用此目录而非默认输出目录
        """
        # 确定输出目录
        output_dir = run_dir if run_dir else self.output_dir
        debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        
        # 构建详细的处理信息
        processing_details = {
            'timestamp': datetime.now().isoformat(),
            'processing_summary': {
                'original_clusters': len(original_clusters),
                'raw_ontologies_extracted': len(raw_ontologies),
                'merged_ontologies': len(merged_ontologies),
                'deduplication_ratio': 1 - (len(merged_ontologies) / max(len(raw_ontologies), 1))
            },
            'stage1_extraction': {
                'clusters_processed': len(original_clusters),
                'ontologies_by_cluster': {
                    str(cluster_id): len([ont for ont in raw_ontologies if ont['source_cluster'] == cluster_id])
                    for cluster_id in original_clusters.keys()
                }
            },
            'stage2_deduplication': {
                'similarity_threshold': Config.SIMILARITY_THRESHOLD,
                'ontologies_before': len(raw_ontologies),
                'ontologies_after': len(merged_ontologies),
                'duplicates_removed': len(raw_ontologies) - len(merged_ontologies)
            },
            'final_ontologies': [
                {
                    'name': str(ont['name']),
                    'description': str(ont['description'])[:200] + '...' if len(str(ont['description'])) > 200 else str(ont['description']),
                    'relationship_count': int(len(ont.get('relationships', []))),
                    'source_clusters': [int(sc) for sc in ont.get('source_clusters', [])],
                    'merged_from_count': int(ont.get('merged_from_count', 1))
                }
                for ont in merged_ontologies
            ]
        }
        
        # 生成文件名
        filename = "ontology_processing_details.json"
        filepath = os.path.join(debug_dir, filename)
        
        # 保存到文件
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(processing_details, f, ensure_ascii=False, indent=2)
            self.logger.info(f"[保存详情] 处理详情已保存: {filepath}")
        except Exception as e:
            self.logger.error(f"[保存详情] 保存失败: {e}")
    
    def _clean_data_for_json(self, data):
        """
        清理数据以确保JSON序列化兼容性，将numpy类型转换为Python原生类型
        
        Args:
            data: 要清理的数据
            
        Returns:
            清理后的数据
        """
        if isinstance(data, dict):
            return {key: self._clean_data_for_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._clean_data_for_json(item) for item in data]
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.str_, np.unicode_)):
            return str(data)
        elif isinstance(data, np.bool_):
            return bool(data)
        else:
            return data