#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
索引缓存管理模块
负责FAISS索引的缓存存储和加载
"""

import logging
import pickle
import faiss
from pathlib import Path
from typing import List, Dict


class IndexCache:
    """索引缓存管理类"""
    
    def __init__(self, cache_dir: str = "cache/indexes"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def _extract_dataset_name(self, corpus_path: Path) -> str:
        """从语料库路径中提取数据集名称"""
        known_datasets = ['musique', 'quality', 'hotpot', 'nq', 'triviaqa', 'test']
        
        # 如果是简单的字符串（测试用），直接返回
        corpus_str = str(corpus_path).lower()
        if corpus_str.startswith('test'):
            return 'test'
        
        parent_name = corpus_path.parent.name.lower()
        
        if parent_name in known_datasets:
            return parent_name
        
        # 从文件名中提取
        filename = corpus_path.stem.lower()
        for dataset in known_datasets:
            if dataset in filename:
                return dataset
        
        # 如果都不匹配，返回一个通用名称
        return 'unknown'
    
    def generate_cache_key(self, corpus_path: str, n_clusters: int, mode: str = "full") -> str:
        """生成缓存键"""
        corpus_path_obj = Path(corpus_path)
        dataset_name = self._extract_dataset_name(corpus_path_obj)
        
        if mode == "documents_only":
            return f"{dataset_name}_documents_only"
        else:
            return f"{dataset_name}_clusters_{n_clusters}"
    
    def get_cache_path(self, cache_key: str) -> Path:
        """获取缓存路径"""
        return self.cache_dir / cache_key
    
    def exists(self, cache_key: str) -> bool:
        """检查缓存是否存在"""
        cache_path = self.get_cache_path(cache_key)
        faiss_path = cache_path / "faiss_index.bin"
        data_path = cache_path / "index_data.pkl"
        return faiss_path.exists() and data_path.exists()
    
    def save(self, cache_key: str, faiss_index, index_documents: List[Dict], 
             documents: List[Dict], summaries: Dict, top_k: int):
        """保存索引到缓存"""
        cache_path = self.get_cache_path(cache_key)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # 保存FAISS索引
        faiss_path = cache_path / "faiss_index.bin"
        faiss.write_index(faiss_index, str(faiss_path))
        
        # 保存相关数据
        data_path = cache_path / "index_data.pkl"
        with open(data_path, 'wb') as f:
            pickle.dump({
                'index_documents': index_documents,
                'documents': documents,
                'summaries': summaries,
                'top_k': top_k
            }, f)
        
        self.logger.info(f"[索引缓存] 索引已保存: {cache_path}")
    
    def load(self, cache_key: str) -> Dict:
        """从缓存加载索引"""
        cache_path = self.get_cache_path(cache_key)
        
        # 加载FAISS索引
        faiss_path = cache_path / "faiss_index.bin"
        faiss_index = faiss.read_index(str(faiss_path))
        
        # 加载相关数据
        data_path = cache_path / "index_data.pkl"
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        data['faiss_index'] = faiss_index
        self.logger.info(f"[索引缓存] 加载完成，包含 {faiss_index.ntotal} 个向量")
        return data