#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本聚类模块
负责对文本片段进行聚类操作
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import json
import os
from datetime import datetime
from .embedding_model import EmbeddingModel

# 尝试导入加速库
try:
    import cuml
    from cuml.cluster import KMeans as CuMLKMeans
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

try:
    from sklearn.utils import parallel_backend
    PARALLEL_BACKEND_AVAILABLE = True
except ImportError:
    PARALLEL_BACKEND_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class TextClusterer:
    """文本聚类器"""
    
    def __init__(self, n_clusters: int = 5, output_dir: str = "results", 
                 min_cluster_size: Optional[int] = None, max_cluster_size: Optional[int] = None,
                 balanced_clustering: bool = False, use_gpu: bool = False, 
                 n_jobs: int = -1, algorithm: str = "auto", run_dir: Optional[str] = None):
        """
        初始化聚类器
        
        Args:
            n_clusters: 聚类数量，应在3-10之间
            output_dir: 输出目录，用于保存聚类详情
            min_cluster_size: 最小聚类大小，None表示不限制
            max_cluster_size: 最大聚类大小，None表示不限制
            balanced_clustering: 是否使用平衡聚类
            use_gpu: 是否使用GPU加速（需要安装cuml）
            n_jobs: CPU并行任务数，-1表示使用所有核心
            algorithm: 聚类算法 ("kmeans", "minibatch", "faiss", "auto")
            run_dir: 运行目录，用于保存聚类详情（优先于output_dir）
        """

        self.n_clusters = n_clusters
        self.output_dir = run_dir if run_dir else output_dir
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.balanced_clustering = balanced_clustering
        self.use_gpu = use_gpu
        self.n_jobs = n_jobs
        self.algorithm = algorithm
        self.logger = logging.getLogger(__name__)
        self.embedding_model = EmbeddingModel()
        self.embeddings = None
        
        # 选择最优的聚类算法实现
        self.kmeans = self._create_kmeans_model()
        
        # 记录加速状态
        self._log_acceleration_status()
    
    def cluster_documents(self, documents: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """
        对文档进行聚类
        
        Args:
            documents: 文档列表
            
        Returns:
            Dict[int, List[Dict]]: 聚类结果，键为聚类ID，值为文档列表
        """
        self.logger.info(f"[聚类] 开始对 {len(documents)} 个文档进行 {self.n_clusters} 聚类")
        
        # 提取文档文本内容
        texts = []
        for doc in documents:
            text = doc.get('context', '') 
            texts.append(text)
        
        # Embedding向量化
        self.logger.info("[聚类] 正在进行文档向量化...")
        self.embeddings = self.embedding_model.encode(texts)
        
        # 聚类算法选择
        if self.balanced_clustering:
            self.logger.info("[聚类] 正在执行加速平衡聚类算法...")
            cluster_labels = self._balanced_kmeans_clustering(self.embeddings)
        else:
            self.logger.info("[聚类] 正在执行加速聚类算法...")
            cluster_labels = self._accelerated_clustering(self.embeddings)
        
        # 组织聚类结果
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(documents[i])
        
        # 记录聚类统计（简化格式）
        cluster_sizes = [f"{cluster_id}({len(docs)})" for cluster_id, docs in sorted(clusters.items())]
        self.logger.info(f"[聚类] 聚类分布: {', '.join(cluster_sizes)}")
        
        # 保存聚类详情
        self._save_clustering_details(documents, clusters, cluster_labels)
        
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
    
    def _save_clustering_details(self, documents: List[Dict[str, Any]], 
                                clusters: Dict[int, List[Dict[str, Any]]], 
                                cluster_labels: np.ndarray):
        """
        保存聚类详情到JSON文件
        
        Args:
            documents: 原始文档列表
            clusters: 聚类结果
            cluster_labels: 聚类标签数组
        """
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 构建详细的聚类信息
        clustering_details = {
            'timestamp': datetime.now().isoformat(),
            'total_documents': len(documents),
            'num_clusters': self.n_clusters,
            'cluster_distribution': {int(k): len(v) for k, v in clusters.items()},
            'clusters': {}
        }
        
        # 为每个聚类保存详细信息
        for cluster_id, cluster_docs in clusters.items():
            cluster_info = {
                'cluster_id': int(cluster_id),
                'document_count': len(cluster_docs),
                'documents': []
            }
            
            # 保存每个文档的详细信息
            for doc in cluster_docs:
                doc_info = {
                    'doc_id': doc.get('id', ''),
                    'title': doc.get('title', ''),
                    'content_preview': doc.get('context', '')[:200] + '...' if len(doc.get('context', '')) > 200 else doc.get('context', ''),
                    'full_content_length': len(doc.get('context', '')),
                    'original_document': doc
                }
                cluster_info['documents'].append(doc_info)
            
            clustering_details['clusters'][int(cluster_id)] = cluster_info
        
        # 创建debug子目录并保存详情文件
        debug_dir = os.path.join(self.output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        
        filename = "clustering_details.json"
        filepath = os.path.join(debug_dir, filename)
        
        # 保存到文件
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(clustering_details, f, ensure_ascii=False, indent=2)
            self.logger.info(f"[聚类] 聚类详情已保存: {filepath}")
        except Exception as e:
            self.logger.error(f"[聚类] 保存聚类详情失败: {e}")
    
    def get_clustering_stats(self, clusters: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        获取聚类统计信息
        
        Args:
            clusters: 聚类结果
            
        Returns:
            Dict: 聚类统计信息
        """
        total_docs = sum(len(docs) for docs in clusters.values())
        cluster_sizes = [len(docs) for docs in clusters.values()]
        
        return {
            'total_documents': total_docs,
            'num_clusters': len(clusters),
            'cluster_sizes': cluster_sizes,
            'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
            'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'avg_cluster_size': sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0
        }
    
    def _balanced_kmeans_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """
        平衡聚类算法：确保每个聚类的大小在指定范围内
        
        Args:
            embeddings: 文档嵌入向量
            
        Returns:
            np.ndarray: 聚类标签
        """
        n_samples = len(embeddings)
        
        # 计算默认的聚类大小限制
        avg_size = n_samples // self.n_clusters
        min_size = self.min_cluster_size or max(1, avg_size // 2)
        max_size = self.max_cluster_size or (avg_size * 3 // 2)
        
        self.logger.info(f"[平衡聚类] 目标大小范围: {min_size}-{max_size} (平均: {avg_size})")
        
        # 第一步：使用加速的KMeans获取初始聚类
        self.logger.info("[平衡聚类] 第一步：使用加速算法进行初始聚类")
        initial_labels = self._accelerated_clustering(embeddings)
        
        # 获取聚类中心（如果可用）
        initial_kmeans = self.kmeans  # 使用已经训练好的模型
        
        # 第二步：重新分配以平衡聚类大小
        balanced_labels = self._rebalance_clusters(embeddings, initial_labels, min_size, max_size)
        
        # 更新kmeans对象以便后续使用
        self.kmeans = initial_kmeans
        
        return balanced_labels
    
    def _rebalance_clusters(self, embeddings: np.ndarray, initial_labels: np.ndarray, 
                           min_size: int, max_size: int) -> np.ndarray:
        """
        重新平衡聚类大小
        
        Args:
            embeddings: 文档嵌入向量
            initial_labels: 初始聚类标签
            min_size: 最小聚类大小
            max_size: 最大聚类大小
            
        Returns:
            np.ndarray: 平衡后的聚类标签
        """
        n_samples = len(embeddings)
        labels = initial_labels.copy()
        
        # 计算当前聚类大小
        cluster_sizes = np.bincount(labels, minlength=self.n_clusters)
        
        self.logger.info(f"[平衡聚类] 初始聚类大小: {cluster_sizes}")
        
        # 迭代重新分配
        max_iterations = 10
        for iteration in range(max_iterations):
            reassigned = False
            
            # 找出过大和过小的聚类
            oversized_clusters = np.where(cluster_sizes > max_size)[0]
            undersized_clusters = np.where(cluster_sizes < min_size)[0]
            
            if len(oversized_clusters) == 0 and len(undersized_clusters) == 0:
                self.logger.info(f"[平衡聚类] 第{iteration+1}轮后达到平衡")
                break
            
            # 从过大的聚类移动文档到过小的聚类
            for oversized_cluster in oversized_clusters:
                if len(undersized_clusters) == 0:
                    break
                
                # 找到属于过大聚类的文档
                oversized_docs = np.where(labels == oversized_cluster)[0]
                
                # 计算这些文档到聚类中心的距离（使用加速计算）
                if hasattr(self.kmeans, 'cluster_centers_'):
                    center = self.kmeans.cluster_centers_[oversized_cluster]
                    distances = self._accelerated_distance_computation(center, embeddings[oversized_docs])
                    # 按距离排序，选择最远的文档进行移动
                    farthest_docs = oversized_docs[np.argsort(distances)[::-1]]
                else:
                    # 如果没有聚类中心，随机选择
                    farthest_docs = oversized_docs
                
                # 移动文档到最需要的聚类
                docs_to_move = min(cluster_sizes[oversized_cluster] - max_size, 
                                 min_size - cluster_sizes[undersized_clusters[0]])
                
                if docs_to_move > 0:
                    target_cluster = undersized_clusters[0]
                    move_indices = farthest_docs[:docs_to_move]
                    
                    # 更新标签
                    labels[move_indices] = target_cluster
                    
                    # 更新聚类大小
                    cluster_sizes[oversized_cluster] -= docs_to_move
                    cluster_sizes[target_cluster] += docs_to_move
                    
                    reassigned = True
                    
                    self.logger.debug(f"[平衡聚类] 移动 {docs_to_move} 个文档: {oversized_cluster} -> {target_cluster}")
                
                # 重新计算过小聚类列表
                undersized_clusters = np.where(cluster_sizes < min_size)[0]
            
            if not reassigned:
                break
        
        final_sizes = np.bincount(labels, minlength=self.n_clusters)
        self.logger.info(f"[平衡聚类] 最终聚类大小: {final_sizes}")
        
        return labels
    
    def _accelerated_distance_computation(self, center: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        加速的距离计算
        
        Args:
            center: 聚类中心向量
            points: 点集合
            
        Returns:
            np.ndarray: 距离数组
        """
        if FAISS_AVAILABLE and len(points) > 1000:
            # 对于大规模数据使用FAISS加速距离计算
            return self._faiss_distance_computation(center, points)
        else:
            # 使用优化的numpy计算
            return self._numpy_distance_computation(center, points)
    
    def _faiss_distance_computation(self, center: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        使用FAISS进行快速距离计算
        
        Args:
            center: 聚类中心向量
            points: 点集合
            
        Returns:
            np.ndarray: 距离数组
        """
        # 确保数据类型正确
        center = center.astype(np.float32).reshape(1, -1)
        points = points.astype(np.float32)
        
        # 创建FAISS索引
        index = faiss.IndexFlatL2(center.shape[1])
        index.add(points)
        
        # 搜索距离
        distances, _ = index.search(center, len(points))
        
        return distances[0]
    
    def _numpy_distance_computation(self, center: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        使用优化的numpy进行距离计算
        
        Args:
            center: 聚类中心向量
            points: 点集合
            
        Returns:
            np.ndarray: 距离数组
        """
        # 使用向量化计算欧几里得距离
        diff = points - center
        distances = np.linalg.norm(diff, axis=1)
        
        return distances
    
    def _create_kmeans_model(self):
        """
        根据配置创建最优的KMeans模型
        
        Returns:
            聚类模型实例
        """
        # GPU加速优先
        if self.use_gpu and CUML_AVAILABLE:
            try:
                self.logger.info("[聚类加速] 使用GPU加速聚类 (cuML)")
                return CuMLKMeans(
                    n_clusters=self.n_clusters,
                    random_state=42,
                    init='k-means++',
                    n_init=10
                )
            except Exception as e:
                self.logger.warning(f"[聚类加速] GPU初始化失败，回退到CPU: {e}")
        
        # 算法选择
        if self.algorithm == "auto":
            # 自动选择最优算法
            algorithm = self._auto_select_algorithm()
        else:
            algorithm = self.algorithm
        
        # 创建对应的模型
        if algorithm == "minibatch":
            self.logger.info("[聚类加速] 使用MiniBatch KMeans")
            return MiniBatchKMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                batch_size=1024,
                n_init=10
            )
        elif algorithm == "faiss" and FAISS_AVAILABLE:
            self.logger.info("[聚类加速] 使用FAISS KMeans")
            return self._create_faiss_kmeans()
        else:
            # 标准KMeans with 多核支持
            self.logger.info(f"[聚类加速] 使用标准KMeans (n_jobs={self.n_jobs})")
            return KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10,
                n_jobs=self.n_jobs
            )
    
    def _auto_select_algorithm(self) -> str:
        """
        根据数据规模自动选择最优算法
        
        Returns:
            str: 算法名称
        """
        # 基于经验的算法选择策略
        if FAISS_AVAILABLE:
            return "faiss"  # FAISS通常最快
        elif hasattr(self, 'embeddings') and self.embeddings is not None:
            n_samples = len(self.embeddings)
            if n_samples > 10000:
                return "minibatch"  # 大数据集使用MiniBatch
            else:
                return "kmeans"  # 小数据集使用标准KMeans
        else:
            return "kmeans"
    
    def _create_faiss_kmeans(self):
        """
        创建FAISS KMeans包装器
        
        Returns:
            FAISS KMeans包装器
        """
        class FAISSKMeans:
            def __init__(self, n_clusters, random_state=42):
                self.n_clusters = n_clusters
                self.random_state = random_state
                self.cluster_centers_ = None
                
            def fit_predict(self, X):
                # 确保数据是float32类型（FAISS要求）
                X = X.astype(np.float32)
                
                # 创建FAISS KMeans
                kmeans = faiss.Kmeans(
                    d=X.shape[1],
                    k=self.n_clusters,
                    niter=20,
                    verbose=False,
                    seed=self.random_state
                )
                
                # 训练并预测
                kmeans.train(X)
                _, labels = kmeans.index.search(X, 1)
                
                # 保存聚类中心
                self.cluster_centers_ = kmeans.centroids
                
                return labels.flatten()
        
        return FAISSKMeans(n_clusters=self.n_clusters, random_state=42)
    
    def _log_acceleration_status(self):
        """记录加速状态"""
        status_info = []
        
        if self.use_gpu:
            if CUML_AVAILABLE:
                status_info.append("GPU加速: ✓ (cuML)")
            else:
                status_info.append("GPU加速: ✗ (需要安装cuml)")
        
        if FAISS_AVAILABLE:
            status_info.append("FAISS加速: ✓")
        else:
            status_info.append("FAISS加速: ✗ (需要安装faiss)")
            
        if PARALLEL_BACKEND_AVAILABLE:
            status_info.append(f"多核并行: ✓ (n_jobs={self.n_jobs})")
        else:
            status_info.append("多核并行: ✗")
        
        status_info.append(f"算法选择: {self.algorithm}")
        
        self.logger.info(f"[聚类加速] {' | '.join(status_info)}")
    
    def _accelerated_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """
        使用加速算法进行聚类
        
        Args:
            embeddings: 文档嵌入向量
            
        Returns:
            np.ndarray: 聚类标签
        """
        self.logger.info(f"[聚类加速] 开始加速聚类，数据形状: {embeddings.shape}")
        
        # 使用并行后端（如果可用）
        if PARALLEL_BACKEND_AVAILABLE and not self.use_gpu:
            with parallel_backend('threading', n_jobs=self.n_jobs):
                labels = self.kmeans.fit_predict(embeddings)
        else:
            labels = self.kmeans.fit_predict(embeddings)
        
        return labels