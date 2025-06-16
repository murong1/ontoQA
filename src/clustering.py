#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本聚类模块
负责对文本片段进行聚类操作
"""

import logging
from typing import List, Dict, Any
from sklearn.cluster import KMeans
import numpy as np
from .embedding_model import EmbeddingModel


class TextClusterer:
    """文本聚类器"""
    
    def __init__(self, n_clusters: int = 5):
        """
        初始化聚类器
        
        Args:
            n_clusters: 聚类数量，应在3-10之间
        """

        self.n_clusters = n_clusters
        self.logger = logging.getLogger(__name__)
        self.embedding_model = EmbeddingModel()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.embeddings = None
    
    def cluster_documents(self, documents: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """
        对文档进行聚类
        
        Args:
            documents: 文档列表
            
        Returns:
            Dict[int, List[Dict]]: 聚类结果，键为聚类ID，值为文档列表
        """
        self.logger.info(f"Starting clustering of {len(documents)} documents into {self.n_clusters} clusters")
        
        # 提取文档文本内容
        texts = []
        for doc in documents:
            text = doc.get('context', '') 
            texts.append(text)
        
        # Embedding向量化
        self.logger.info("Performing embedding vectorization...")
        self.embeddings = self.embedding_model.encode(texts)
        
        # K-means聚类
        self.logger.info("Performing K-means clustering...")
        cluster_labels = self.kmeans.fit_predict(self.embeddings)
        
        # 组织聚类结果
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(documents[i])
        
        # 记录聚类统计
        for cluster_id, docs in clusters.items():
            self.logger.info(f"Cluster {cluster_id}: {len(docs)} documents")
        
        return clusters
    
    def get_cluster_center(self, cluster_id: int) -> np.ndarray:
        """
        获取聚类中心向量
        
        Args:
            cluster_id: 聚类ID
            
        Returns:
            np.ndarray: 聚类中心向量
        """
        if not hasattr(self.kmeans, 'cluster_centers_'):
            raise ValueError("Model has not been fitted yet")
        
        return self.kmeans.cluster_centers_[cluster_id]
    
    def get_cluster_similarity(self, cluster_id: int, document_texts: List[str]) -> List[float]:
        """
        计算文档与聚类中心的相似度
        
        Args:
            cluster_id: 聚类ID
            document_texts: 文档文本列表
            
        Returns:
            List[float]: 相似度列表
        """
        if not hasattr(self.kmeans, 'cluster_centers_'):
            raise ValueError("Model has not been fitted yet")
        
        # 获取聚类中心
        center = self.get_cluster_center(cluster_id)
        
        # 计算文档embedding
        doc_embeddings = self.embedding_model.encode(document_texts)
        
        # 计算余弦相似度
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([center], doc_embeddings)[0]
        
        return similarities.tolist()