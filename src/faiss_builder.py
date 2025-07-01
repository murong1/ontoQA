#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAISS索引构建模块
负责构建和管理FAISS向量索引
"""

import logging
import faiss
import numpy as np
from typing import List


class FaissIndexBuilder:
    """FAISS索引构建器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        """构建FAISS索引"""
        embeddings = embeddings.astype('float32')
        dimension = embeddings.shape[1]
        
        # 使用内积搜索，L2归一化后等同于余弦相似度
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        self.logger.info(f"[FAISS索引] 索引创建完成，{index.ntotal} 个向量，维度: {dimension}")
        return index
    
    def search(self, index: faiss.Index, query_embedding: np.ndarray, top_k: int):
        """在索引中搜索相似向量"""
        return index.search(query_embedding.astype('float32'), top_k)