#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本体总结模块
负责使用LLM对聚类结果进行两阶段本体处理：
1. 从聚类生成结构化本体
2. 跨聚类本体去重和合并
"""

import logging
from typing import Dict, List, Any
from collections import defaultdict
from .ontology_extractor import OntologyExtractor
from .ontology_deduplicator import OntologyDeduplicator
from .ontology_merger import OntologyMerger
from .cache_manager import CacheManager
from .ontology_config import OntologyConfig


class OntologySummarizer:
    """两阶段本体处理器（模块化重构版本）"""
    
    def __init__(self, output_dir: str = "results"):
        """
        初始化总结器
        
        Args:
            output_dir: 输出目录，用于保存本体生成详情
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        
        # 初始化各个组件
        self.extractor = OntologyExtractor()
        self.deduplicator = OntologyDeduplicator()
        self.merger = OntologyMerger()
        self.cache_manager = CacheManager(output_dir)
    
    def summarize_clusters(self, clusters: Dict[int, List[Dict[str, Any]]]) -> Dict[int, Dict[str, Any]]:
        """
        两阶段本体处理主入口
        
        Args:
            clusters: 聚类结果
            
        Returns:
            Dict[int, Dict]: 最终本体集合，与现有系统兼容
        """
        # 验证输入
        if not clusters:
            raise ValueError("聚类数据为空，无法进行本体处理")
        
        if not all(isinstance(docs, list) and docs for docs in clusters.values()):
            raise ValueError("聚类数据包含空的文档列表")
        
        self.logger.info(f"[本体处理] 开始两阶段处理 {len(clusters)} 个聚类")
        
        # 生成唯一标识符用于缓存
        cluster_signature = self.cache_manager.generate_cluster_signature(clusters)
        
        # 第一阶段：从聚类抽取结构化本体（带缓存）
        self.logger.info("[本体处理] 第一阶段：抽取结构化本体")
        raw_ontologies = self._extract_with_cache(clusters, cluster_signature)
        
        # 第二阶段：本体去重和合并（带缓存）
        self.logger.info("[本体处理] 第二阶段：本体去重和合并")
        merged_ontologies = self._merge_with_cache(raw_ontologies, cluster_signature)
        
        # 转换为兼容格式
        self.logger.info("[本体处理] 转换为兼容格式")
        compatible_result = self.merger.convert_to_compatible_format(merged_ontologies)
        
        # 保存处理详情
        self.cache_manager.save_processing_details(clusters, raw_ontologies, merged_ontologies)
        
        self.logger.info(f"[本体处理] 完成处理，最终得到 {len(merged_ontologies)} 个独特本体")
        
        return compatible_result
    
    def _extract_with_cache(self, clusters: Dict[int, List[Dict[str, Any]]], signature: str) -> List[Dict[str, Any]]:
        """
        带缓存的第一阶段本体抽取
        
        Args:
            clusters: 聚类数据
            signature: 数据签名
            
        Returns:
            List[Dict]: 原始本体列表
        """
        # 尝试从缓存加载
        cached_ontologies = self.cache_manager.load_extraction_cache(signature)
        if cached_ontologies is not None:
            return cached_ontologies
        
        # 缓存不存在，执行抽取
        self.logger.info("[缓存] 第一阶段缓存不存在，开始计算")
        raw_ontologies = self.extractor.extract_from_clusters(clusters)
        
        # 保存到缓存
        self.cache_manager.save_extraction_cache(signature, raw_ontologies)
        
        return raw_ontologies
    
    def _merge_with_cache(self, raw_ontologies: List[Dict[str, Any]], signature: str) -> List[Dict[str, Any]]:
        """
        带缓存的第二阶段本体去重和合并
        
        Args:
            raw_ontologies: 原始本体列表
            signature: 数据签名
            
        Returns:
            List[Dict]: 合并后的本体列表
        """
        # 尝试从缓存加载
        cached_ontologies = self.cache_manager.load_merge_cache(signature)
        if cached_ontologies is not None:
            return cached_ontologies
        
        # 缓存不存在，执行去重和合并
        self.logger.info("[缓存] 第二阶段缓存不存在，开始计算")
        
        # 使用去重器进行去重
        ontology_groups = self.deduplicator.deduplicate(raw_ontologies)
        
        # 使用合并器进行合并
        merged_ontologies = self.merger.merge_ontology_groups(ontology_groups)
        
        # 保存到缓存
        self.cache_manager.save_merge_cache(signature, merged_ontologies)
        
        return merged_ontologies
    
    def _legacy_extract_from_clusters(self, clusters: Dict[int, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        遗留方法：从聚类中抽取结构化本体（保留向后兼容性）
        注意：建议使用新的模块化接口
        
        Args:
            clusters: 聚类结果
            
        Returns:
            List[Dict]: 原始本体列表
        """
        # 使用新的模块化接口
        return self.extractor.extract_from_clusters(clusters)
    
    def _legacy_process_single_cluster(self, cluster_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        遗留方法：处理单个聚类的本体抽取（保留向后兼容性）
        注意：建议使用新的模块化接口
        
        Args:
            cluster_info: 聚类信息
            
        Returns:
            List[Dict]: 本体列表
        """
        # 重定向到新的抽取器
        cluster_id = cluster_info['cluster_id']
        documents = cluster_info['documents']
        clusters = {cluster_id: documents}
        return self.extractor.extract_from_clusters(clusters)
    
    def _legacy_build_prompt(self, contexts: List[str], titles: List[str]) -> str:
        """
        遗留方法：构建本体抽取提示词（保留向后兼容性）
        注意：建议使用PromptManager
        
        Args:
            contexts: 文档内容列表
            titles: 文档标题列表
            
        Returns:
            str: 构建的提示词
        """
        from .prompt_manager import PromptManager
        return PromptManager.build_ontology_extraction_prompt(contexts, titles)
    
    def _legacy_parse_response(self, llm_response: str, source_cluster: int) -> List[Dict[str, Any]]:
        """
        遗留方法：解析LLM响应（保留向后兼容性）
        注意：建议使用OntologyExtractor的解析方法
        
        Args:
            llm_response: LLM响应
            source_cluster: 来源聚类ID
            
        Returns:
            List[Dict]: 本体列表
        """
        # 重定向到抽取器的解析方法
        return self.extractor._parse_ontologies_from_response(llm_response, source_cluster)
    
    def _legacy_fallback_parse(self, llm_response: str, source_cluster: int) -> List[Dict[str, Any]]:
        """
        遗留方法：回退解析（保留向后兼容性）
        注意：建议使用OntologyExtractor的解析方法
        
        Args:
            llm_response: LLM响应
            source_cluster: 来源聚类ID
            
        Returns:
            List[Dict]: 本体列表
        """
        # 重定向到抽取器的回退解析方法
        return self.extractor._fallback_parse_ontologies(llm_response, source_cluster)
    
    def _legacy_deduplicate_and_merge(self, ontologies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        遗留方法：本体去重和合并（保留向后兼容性）
        注意：建议使用新的模块化接口
        
        Args:
            ontologies: 原始本体列表
            
        Returns:
            List[Dict]: 去重后的本体列表
        """
        # 使用新的模块化接口
        ontology_groups = self.deduplicator.deduplicate(ontologies)
        return self.merger.merge_ontology_groups(ontology_groups)
    
    def _legacy_merge_by_name(self, ontologies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        遗留方法：按名称合并本体（保留向后兼容性）
        注意：建议使用OntologyDeduplicator
        
        Args:
            ontologies: 本体列表
            
        Returns:
            List[Dict]: 合并后的本体列表
        """
        # 重定向到去重器
        groups = self.deduplicator._merge_by_exact_name(ontologies)
        # 扁平化为单个本体列表以保持向后兼容
        merged_ontologies = []
        for group in groups:
            if len(group) == 1:
                merged_ontologies.extend(group)
            else:
                merged_ontology = self.merger._merge_single_group(group)
                merged_ontologies.append(merged_ontology)
        return merged_ontologies
    
    def _legacy_semantic_clustering(self, ontologies: List[Dict[str, Any]], 
                                   embeddings) -> List[List[Dict[str, Any]]]:
        """
        遗留方法：语义相似度聚类（保留向后兼容性）
        注意：建议使用OntologyDeduplicator
        
        Args:
            ontologies: 本体列表
            embeddings: 嵌入向量
            
        Returns:
            List[List[Dict]]: 聚类结果
        """
        # 重定向到去重器
        # 先将本体转为组格式（每个本体一组）
        ontology_groups = [[ont] for ont in ontologies]
        return self.deduplicator._semantic_similarity_clustering(ontology_groups, embeddings)
    
    def _legacy_merge_semantic_groups(self, semantic_groups: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        遗留方法：合并语义组（保留向后兼容性）
        注意：建议使用OntologyMerger
        
        Args:
            semantic_groups: 语义聚类结果
            
        Returns:
            List[Dict]: 合并后的本体列表
        """
        # 重定向到合并器
        return self.merger.merge_ontology_groups(semantic_groups)
    
    def _legacy_compute_embeddings(self, texts: List[str], batch_size: int = 50):
        """
        遗留方法：并发计算嵌入向量（保留向后兼容性）
        注意：建议使用OntologyDeduplicator
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            嵌入向量数组
        """
        # 重定向到去重器
        return self.deduplicator._compute_embeddings_concurrent(texts)
    
    def _legacy_get_embedding_dim(self) -> int:
        """
        遗留方法：获取嵌入维度（保留向后兼容性）
        
        Returns:
            int: 嵌入向量维度
        """
        try:
            from .embedding_model import EmbeddingModel
            embedding_model = EmbeddingModel()
            test_embedding = embedding_model.encode_batch(["test"])
            return test_embedding.shape[1]
        except Exception as e:
            self.logger.error(f"[嵌入计算] 无法获取嵌入维度: {e}")
            raise RuntimeError(f"无法获取嵌入模型维度: {e}") from e
    
    def _legacy_compute_similarity_matrix(self, embeddings, batch_size: int = 1000):
        """
        遗留方法：分批计算相似度矩阵（保留向后兼容性）
        注意：当前实现忽略内存优化要求
        
        Args:
            embeddings: 嵌入向量
            batch_size: 批处理大小
            
        Returns:
            相似度矩阵
        """
        import numpy as np
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))
        
        self.logger.info(f"[去重] 分批计算 {n}x{n} 相似度矩阵")
        
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            
            for j in range(i, n, batch_size):
                end_j = min(j + batch_size, n)
                
                # 计算批次间的余弦相似度
                batch_similarities = np.dot(embeddings[i:end_i], embeddings[j:end_j].T)
                
                # 标准化（假设embeddings已经归一化）
                similarity_matrix[i:end_i, j:end_j] = batch_similarities
                
                # 对称填充
                if i != j:
                    similarity_matrix[j:end_j, i:end_i] = batch_similarities.T
        
        return similarity_matrix
    
    def _legacy_calculate_similarity(self, emb1, emb2) -> float:
        """
        遗留方法：计算余弦相似度（保留向后兼容性）
        注意：建议使用OntologyDeduplicator
        
        Args:
            emb1: 第一个向量
            emb2: 第二个向量
            
        Returns:
            float: 余弦相似度
        """
        # 重定向到去重器
        return self.deduplicator._calculate_similarity(emb1, emb2)
    
    def _legacy_merge_group(self, ontology_group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        遗留方法：合并本体组（保留向后兼容性）
        注意：建议使用OntologyMerger
        
        Args:
            ontology_group: 重复本体组
            
        Returns:
            Dict: 合并后的本体
        """
        # 重定向到合并器
        return self.merger._merge_single_group(ontology_group)
    
    def _legacy_llm_merge_info(self, ontology_name: str, descriptions: List[str], 
                              relationships: List[str]) -> Dict[str, Any]:
        """
        遗留方法：LLM合并本体信息（保留向后兼容性）
        注意：建议使用OntologyMerger
        
        Args:
            ontology_name: 本体名称
            descriptions: 描述列表
            relationships: 关系列表
            
        Returns:
            Dict: 包含description和relationships
        """
        # 重定向到合并器
        return self.merger._merge_ontology_info(ontology_name, descriptions, relationships)
    
    def _legacy_merge_groups_concurrent(self, duplicate_groups: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        遗留方法：并发合并本体组（保留向后兼容性）
        注意：建议使用OntologyMerger
        
        Args:
            duplicate_groups: 重复本体组列表
            
        Returns:
            List[Dict]: 合并后的本体列表
        """
        # 重定向到合并器
        return self.merger._merge_groups_concurrent(duplicate_groups)
    
    def _legacy_build_merge_prompt(self, ontology_name: str, descriptions: List[str], 
                                  relationships: List[str]) -> str:
        """
        遗留方法：构建合并提示词（保留向后兼容性）
        注意：建议使用PromptManager
        
        Args:
            ontology_name: 本体名称
            descriptions: 描述列表
            relationships: 关系列表
            
        Returns:
            str: 合并提示词
        """
        from .prompt_manager import PromptManager
        return PromptManager.build_merge_prompt(ontology_name, descriptions, relationships)
    
    def _legacy_parse_merge_response(self, response: str) -> Dict[str, Any]:
        """
        遗留方法：解析合并响应（保留向后兼容性）
        注意：建议使用OntologyMerger
        
        Args:
            response: LLM响应
            
        Returns:
            Dict: 包含description和relationships
        """
        # 重定向到合并器
        return self.merger._parse_merge_response(response)
    
    def _legacy_convert_format(self, merged_ontologies: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        遗留方法：转换为兼容格式（保留向后兼容性）
        注意：建议使用OntologyMerger
        
        Args:
            merged_ontologies: 合并后的本体列表
            
        Returns:
            Dict[int, Dict]: 兼容格式的结果
        """
        # 重定向到合并器
        return self.merger.convert_to_compatible_format(merged_ontologies)
    
    def _legacy_save_details(self, original_clusters: Dict[int, List[Dict[str, Any]]], 
                            raw_ontologies: List[Dict[str, Any]], 
                            merged_ontologies: List[Dict[str, Any]]) -> None:
        """
        遗留方法：保存处理详情（保留向后兼容性）
        注意：建议使用CacheManager
        
        Args:
            original_clusters: 原始聚类结果
            raw_ontologies: 原始本体列表
            merged_ontologies: 合并后的本体列表
        """
        # 重定向到缓存管理器
        self.cache_manager.save_processing_details(original_clusters, raw_ontologies, merged_ontologies)
    
    def get_ontology_stats(self, summaries: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        获取本体处理统计信息
        
        Args:
            summaries: 处理结果
            
        Returns:
            Dict: 统计信息
        """
        if not summaries:
            return {}
        
        total_ontologies = len(summaries)
        total_relationships = sum(len(s.get('relationships', [])) for s in summaries.values())
        
        # 统计来源聚类分布
        source_cluster_counts = defaultdict(int)
        for summary in summaries.values():
            for cluster_id in summary.get('source_clusters', []):
                source_cluster_counts[cluster_id] += 1
        
        return {
            'total_ontologies': total_ontologies,
            'total_relationships': total_relationships,
            'avg_relationships_per_ontology': total_relationships / total_ontologies if total_ontologies > 0 else 0,
            'source_cluster_distribution': dict(source_cluster_counts),
            'processing_method': 'modular_two_stage_deduplication'
        }
    
    def _legacy_generate_signature(self, clusters: Dict[int, List[Dict[str, Any]]]) -> str:
        """
        遗留方法：生成聚类签名（保留向后兼容性）
        注意：建议使用CacheManager
        
        Args:
            clusters: 聚类数据
            
        Returns:
            str: 唯一签名字符串
        """
        # 重定向到缓存管理器
        return self.cache_manager.generate_cluster_signature(clusters)
    
    def _legacy_extract_with_cache(self, clusters: Dict[int, List[Dict[str, Any]]], 
                                  signature: str) -> List[Dict[str, Any]]:
        """
        遗留方法：带缓存的抽取（保留向后兼容性）
        注意：建议使用新的_extract_with_cache方法
        
        Args:
            clusters: 聚类数据
            signature: 数据签名
            
        Returns:
            List[Dict]: 原始本体列表
        """
        # 重定向到新方法
        return self._extract_with_cache(clusters, signature)
    
    def _legacy_merge_with_cache(self, raw_ontologies: List[Dict[str, Any]], 
                                signature: str) -> List[Dict[str, Any]]:
        """
        遗留方法：带缓存的合并（保留向后兼容性）
        注意：建议使用新的_merge_with_cache方法
        
        Args:
            raw_ontologies: 原始本体列表
            signature: 数据签名
            
        Returns:
            List[Dict]: 去重后的本体列表
        """
        # 重定向到新方法
        return self._merge_with_cache(raw_ontologies, signature)
    
    def _legacy_clean_data(self, data):
        """
        遗留方法：清理数据（保留向后兼容性）
        注意：建议使用CacheManager
        
        Args:
            data: 要清理的数据
            
        Returns:
            清理后的数据
        """
        # 重定向到缓存管理器
        return self.cache_manager._clean_data_for_json(data)