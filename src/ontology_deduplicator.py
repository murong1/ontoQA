#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本体去重模块
负责对本体进行去重处理
"""

import logging
import concurrent.futures
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict
from tqdm import tqdm
from .embedding_model import EmbeddingModel
from config import Config


class OntologyDeduplicator:
    """本体去重器"""
    
    def __init__(self):
        """初始化本体去重器"""
        self.logger = logging.getLogger(__name__)
        self.embedding_model = EmbeddingModel()
    
    def deduplicate(self, ontologies: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        对本体进行去重处理，返回去重后的本体组
        
        Args:
            ontologies: 本体列表
            
        Returns:
            List[List[Dict]]: 去重后的本体组列表
        """
        if not ontologies:
            return []
        
        self.logger.info(f"[本体去重] 开始两步法去重处理 {len(ontologies)} 个本体")
        
        # 第一步：按名称合并同名本体
        name_merged_groups = self._merge_by_exact_name(ontologies)
        self.logger.info(f"[本体去重] 名称合并完成：{len(ontologies)} -> {len(name_merged_groups)} 个本体组")
        
        # 第二步：对合并后的本体进行语义相似度去重
        if len(name_merged_groups) <= 1:
            self.logger.info(f"[本体去重] 只有 {len(name_merged_groups)} 个本体组，跳过语义去重")
            return name_merged_groups
        
        self.logger.info(f"[本体去重] 开始语义相似度去重，处理 {len(name_merged_groups)} 个本体组")
        
        # 计算本体组代表的嵌入向量
        try:
            representative_names = [group[0]["name"] for group in name_merged_groups]
            name_embeddings = self._compute_embeddings_concurrent(representative_names)
            self.logger.info(f"[本体去重] 计算了 {len(name_embeddings)} 个本体组的嵌入向量")
        except Exception as e:
            self.logger.error(f"[本体去重] 计算嵌入向量失败: {e}")
            raise RuntimeError(f"嵌入向量计算失败: {e}") from e
        
        # 语义相似度聚类
        semantic_groups = self._semantic_similarity_clustering(name_merged_groups, name_embeddings)
        
        self.logger.info(f"[本体去重] 两步法去重完成：{len(ontologies)} -> {len(semantic_groups)} 个本体组")
        
        return semantic_groups
    
    def _merge_by_exact_name(self, ontologies: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        按精确名称合并同名本体
        
        Args:
            ontologies: 本体列表
            
        Returns:
            List[List[Dict]]: 按名称分组的本体列表
        """
        name_groups = defaultdict(list)
        
        # 按名称分组
        for ont in ontologies:
            normalized_name = ont["name"].lower().strip()
            name_groups[normalized_name].append(ont)
        
        merged_groups = []
        duplicate_count = 0
        
        for name, group in name_groups.items():
            if len(group) > 1:
                duplicate_count += len(group)
                self.logger.debug(f"[名称合并] 发现 {len(group)} 个同名本体: {group[0]['name']}")
            
            merged_groups.append(group)
        
        if duplicate_count > 0:
            self.logger.info(f"[名称合并] 发现 {duplicate_count} 个同名本体")
        
        return merged_groups
    
    def _semantic_similarity_clustering(self, ontology_groups: List[List[Dict[str, Any]]], 
                                       embeddings: np.ndarray) -> List[List[Dict[str, Any]]]:
        """
        基于语义相似度对本体组进行聚类
        
        Args:
            ontology_groups: 本体组列表
            embeddings: 对应的嵌入向量
            
        Returns:
            List[List[Dict]]: 语义聚类结果
        """
        n = len(ontology_groups)
        visited = [False] * n
        semantic_groups = []
        
        self.logger.info(f"[语义聚类] 对 {n} 个本体组进行语义相似度聚类")
        
        for i in range(n):
            if visited[i]:
                continue
            
            # 创建新的语义组，包含所有相似的本体
            current_semantic_group = []
            current_semantic_group.extend(ontology_groups[i])  # 添加第i组的所有本体
            visited[i] = True
            
            # 计算与所有未访问本体组的相似度
            for j in range(i + 1, n):
                if visited[j]:
                    continue
                
                similarity = self._calculate_similarity(embeddings[i], embeddings[j])
                
                if similarity > Config.SIMILARITY_THRESHOLD:
                    current_semantic_group.extend(ontology_groups[j])  # 添加第j组的所有本体
                    visited[j] = True
                    self.logger.debug(f"[语义聚类] 发现相似本体组: {ontology_groups[i][0]['name']} <-> {ontology_groups[j][0]['name']} (相似度: {similarity:.3f})")
            
            semantic_groups.append(current_semantic_group)
        
        semantic_duplicates = len([g for g in semantic_groups if len(g) > 1])
        total_semantic_duplicates = sum(len(g) for g in semantic_groups if len(g) > 1)
        
        self.logger.info(f"[语义聚类] 发现 {semantic_duplicates} 个语义相似组，涉及 {total_semantic_duplicates} 个本体")
        
        return semantic_groups
    
    def _compute_embeddings_concurrent(self, texts: List[str]) -> np.ndarray:
        """
        并发计算文本嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            np.ndarray: 嵌入向量数组
        """
        batch_size = Config.EMBEDDING_BATCH_SIZE
        
        if len(texts) <= batch_size:
            return self.embedding_model.encode_batch(texts, verbose=False)
        
        self.logger.info(f"[嵌入计算] 开始并发计算 {len(texts)} 个文本的嵌入向量")
        
        # 分批处理
        text_batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        all_embeddings = []
        
        # 使用线程池并发处理批次
        max_workers = min(len(text_batches), Config.MAX_CONCURRENT_EMBEDDING)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有批次任务
            future_to_batch = {
                executor.submit(self.embedding_model.encode_batch, batch, False): i
                for i, batch in enumerate(text_batches)
            }
            
            # 使用进度条收集结果（保持顺序）
            batch_results = [None] * len(text_batches)
            with tqdm(total=len(text_batches), desc="本体向量化", unit="批次", position=2, leave=False) as pbar:
                try:
                    for future in concurrent.futures.as_completed(future_to_batch):
                        batch_idx = future_to_batch[future]
                        try:
                            batch_embeddings = future.result()
                            batch_results[batch_idx] = batch_embeddings
                            pbar.update(1)
                            pbar.set_postfix({"批次": f"{batch_idx + 1}/{len(text_batches)}"})
                        except Exception as e:
                            tqdm.write(f"[ERROR] [嵌入计算] 批次 {batch_idx + 1} 失败: {e}")
                            pbar.update(1)
                            raise RuntimeError(f"嵌入计算批次 {batch_idx + 1} 失败: {e}") from e
                except KeyboardInterrupt:
                    pbar.close()
                    tqdm.write("[WARNING] [嵌入计算] 用户中断操作")
                    # 取消所有未完成的任务
                    for future in future_to_batch.keys():
                        future.cancel()
                    raise
        
        # 合并所有批次结果
        for batch_result in batch_results:
            if batch_result is not None:
                all_embeddings.append(batch_result)
        
        if all_embeddings:
            combined_embeddings = np.vstack(all_embeddings)
            self.logger.info(f"[嵌入计算] 并发计算完成，得到 {len(combined_embeddings)} 个向量")
            return combined_embeddings
        else:
            raise RuntimeError("所有嵌入计算批次都失败")
    
    def _calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        计算两个嵌入向量的余弦相似度
        
        Args:
            emb1: 第一个向量
            emb2: 第二个向量
            
        Returns:
            float: 余弦相似度
        """
        try:
            # 计算余弦相似度
            dot_product = np.dot(emb1, emb2)
            norm_a = np.linalg.norm(emb1)
            norm_b = np.linalg.norm(emb2)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            similarity = dot_product / (norm_a * norm_b)
            return float(similarity)
        
        except Exception as e:
            self.logger.error(f"[相似度计算] 失败: {e}")
            return 0.0