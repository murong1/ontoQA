#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本体总结模块
负责使用LLM对聚类结果进行两阶段本体处理：
1. 从聚类生成结构化本体
2. 跨聚类本体去重和合并
"""

import logging
from typing import List, Dict, Any
import json
import os
import re
import hashlib
from datetime import datetime
from .llm_model import LLMModel
from .embedding_model import EmbeddingModel
import numpy as np
from collections import defaultdict
import concurrent.futures
import threading


class OntologySummarizer:
    """两阶段本体处理器（并发优化版本）"""
    
    def __init__(self, output_dir: str = "results", max_concurrent_llm: int = 200, max_concurrent_embedding: int = 3):
        """
        初始化总结器
        
        Args:
            output_dir: 输出目录，用于保存本体生成详情
            max_concurrent_llm: LLM调用的最大并发数
            max_concurrent_embedding: 嵌入计算的最大并发数
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        self.llm_model = LLMModel(llm_type="summary")
        self.embedding_model = EmbeddingModel()
        
        # 本体去重的相似度阈值
        self.similarity_threshold = 0.85
        
        # 并发控制参数
        self.max_concurrent_llm = max_concurrent_llm
        self.max_concurrent_embedding = max_concurrent_embedding
        
        # 线程安全锁
        self._processing_lock = threading.Lock()
    
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
        
        # 生成唯一标识符用于中间文件
        cluster_signature = self._generate_cluster_signature(clusters)
        
        # 第一阶段：从聚类抽取结构化本体
        self.logger.info("[本体处理] 第一阶段：抽取结构化本体")
        raw_ontologies = self._extract_ontologies_from_clusters_with_cache(clusters, cluster_signature)
        
        # 第二阶段：本体去重和合并
        self.logger.info("[本体处理] 第二阶段：本体去重和合并")
        merged_ontologies = self._deduplicate_and_merge_ontologies_with_cache(raw_ontologies, cluster_signature)
        
        # 转换为兼容格式
        self.logger.info("[本体处理] 转换为兼容格式")
        compatible_result = self._convert_to_compatible_format(merged_ontologies)
        
        # 保存处理详情
        self._save_ontology_details(clusters, raw_ontologies, merged_ontologies)
        
        self.logger.info(f"[本体处理] 完成处理，最终得到 {len(merged_ontologies)} 个独特本体")
        
        return compatible_result
    
    def _extract_ontologies_from_clusters(self, clusters: Dict[int, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        第一阶段：从聚类中抽取结构化本体（并发优化版本）
        
        Args:
            clusters: 聚类结果
            
        Returns:
            List[Dict]: 原始本体列表，每个本体包含name, description, relationships, source_cluster
        """
        self.logger.info(f"[第一阶段] 开始并发处理 {len(clusters)} 个聚类的本体抽取")
        
        # 准备并发处理数据
        cluster_data = []
        for cluster_id, documents in clusters.items():
            self.logger.info(f"[第一阶段] 准备聚类 {cluster_id}，包含 {len(documents)} 个文档")
            
            # 提取文档内容
            contexts = []
            titles = []
            for doc in documents:
                contexts.append(doc.get('context', ''))
                titles.append(doc.get('title', ''))
            
            # 构建结构化提示词
            prompt = self._build_ontology_extraction_prompt(contexts, titles)
            
            cluster_data.append({
                'cluster_id': cluster_id,
                'documents': documents,
                'prompt': prompt
            })
        
        # 智能选择处理策略
        self.logger.info(f"[第一阶段] 开始智能处理 {len(cluster_data)} 个聚类")
        all_ontologies = []
        
        # 根据聚类数量智能选择处理策略
        if len(cluster_data) <= 2:
            # 小规模数据：直接顺序处理
            self.logger.info(f"[第一阶段] 小规模数据，使用顺序处理")
            for cluster_info in cluster_data:
                ontologies = self._process_single_cluster_extraction(cluster_info)
                all_ontologies.extend(ontologies)
                cluster_id = cluster_info['cluster_id']
                self.logger.info(f"[第一阶段] 聚类 {cluster_id} 抽取到 {len(ontologies)} 个本体")
        else:
            # 大规模数据：使用并发处理
            max_workers = min(len(cluster_data), self.max_concurrent_llm)
            self.logger.info(f"[第一阶段] 大规模数据，使用并发处理 (max_workers={max_workers})")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_cluster = {
                    executor.submit(self._process_single_cluster_extraction, cluster_info): cluster_info
                    for cluster_info in cluster_data
                }
                
                # 收集结果
                for future in concurrent.futures.as_completed(future_to_cluster):
                    cluster_info = future_to_cluster[future]
                    cluster_id = cluster_info['cluster_id']
                    
                    try:
                        ontologies = future.result()
                        all_ontologies.extend(ontologies)
                        self.logger.info(f"[第一阶段] 聚类 {cluster_id} 抽取到 {len(ontologies)} 个本体")
                    except Exception as e:
                        self.logger.error(f"[第一阶段] 聚类 {cluster_id} 处理失败: {e}")
                        raise RuntimeError(f"聚类 {cluster_id} 本体抽取失败: {e}") from e
        
        self.logger.info(f"[第一阶段] 并发处理完成，总共抽取到 {len(all_ontologies)} 个原始本体")
        return all_ontologies
    
    def _process_single_cluster_extraction(self, cluster_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        处理单个聚类的本体抽取（线程安全，带重试逻辑）
        
        Args:
            cluster_info: 聚类信息，包含cluster_id, documents, prompt
            
        Returns:
            List[Dict]: 从该聚类提取的本体列表
        """
        cluster_id = cluster_info['cluster_id']
        prompt = cluster_info['prompt']
        
        max_retries = 20  # 最大重试次数
        
        for attempt in range(max_retries):
            try:
                # 调用LLM进行本体抽取
                llm_response = self.llm_model.generate_single(prompt)
                
                # 解析响应
                ontologies = self._parse_ontologies_from_response(llm_response, cluster_id)
                
                if ontologies:  # 成功解析到本体
                    if attempt > 0:
                        self.logger.info(f"[单聚类处理] 聚类 {cluster_id} 第 {attempt + 1} 次尝试成功")
                    return ontologies
                else:
                    self.logger.warning(f"[单聚类处理] 聚类 {cluster_id} 第 {attempt + 1} 次尝试未解析到本体")
                    
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.warning(f"[单聚类处理] 聚类 {cluster_id} 第 {attempt + 1} 次尝试JSON解析失败: {e}")
                if attempt == max_retries - 1:
                    self.logger.error(f"[单聚类处理] 聚类 {cluster_id} 达到最大重试次数，JSON解析持续失败")
                    raise RuntimeError(f"聚类 {cluster_id} JSON解析持续失败: {e}") from e
                continue
                
            except Exception as e:
                self.logger.error(f"[单聚类处理] 聚类 {cluster_id} 第 {attempt + 1} 次尝试失败: {e}")
                if attempt == max_retries - 1:
                    raise RuntimeError(f"聚类 {cluster_id} 处理失败: {e}") from e
                continue
        
        # 如果所有重试都失败
        raise RuntimeError(f"聚类 {cluster_id} 经过 {max_retries} 次重试仍然失败")
    
    def _build_ontology_extraction_prompt(self, contexts: List[str], titles: List[str]) -> str:
        """
        构建本体抽取的结构化提示词
        
        Args:
            contexts: 文档内容列表
            titles: 文档标题列表
            
        Returns:
            str: 构建的提示词
        """
        documents_text = "\n\n".join([
            f"Document {i+1}:\nTitle: {title}\nContent: {context}" 
            for i, (title, context) in enumerate(zip(titles, contexts))
        ])
        
        prompt = f"""Extract independent ontological concepts from the following document cluster. Each ontology should be a clear, independently describable conceptual entity.

Requirements:
1. Identify the most important **independent ontological concepts** in the cluster
2. Provide concise and complete descriptions for each ontology
3. Extract relationships between ontologies, which can connect main ontologies and other concepts
4. Strictly follow JSON format for output

Output format example:
```json
{{
  "ontologies": [
    {{
      "name": "Machine Learning",
      "description": "An artificial intelligence technology that enables computer systems to automatically learn and improve through data training algorithms.",
      "relationships": [
        "Machine Learning -> contains -> Supervised Learning",
        "Machine Learning -> applies to -> Data Mining",
        "Algorithm -> used in -> Machine Learning"
      ]
    }},
    {{
      "name": "Supervised Learning",
      "description": "A branch of machine learning that uses labeled training data to learn mappings from inputs to outputs.",
      "relationships": [
        "Supervised Learning -> is type of -> Machine Learning",
        "Training Data -> input to -> Supervised Learning"
      ]
    }}
  ]
}}
```

Notes:
- Ontology names should be concise and clear
- Descriptions should be self-contained, not dependent on other ontology definitions
- Use "Concept A -> Relationship Type -> Concept B" format for relationships
- Relationships can include ontology names or other relevant concepts
- Ensure JSON format is correct and parseable by programs

Document Cluster:
{documents_text}

Please extract independent ontological concepts:"""
        
        return prompt
    
    def _parse_ontologies_from_response(self, llm_response: str, source_cluster: int) -> List[Dict[str, Any]]:
        """
        从LLM响应中解析本体
        
        Args:
            llm_response: LLM的响应文本
            source_cluster: 来源聚类ID
            
        Returns:
            List[Dict]: 解析出的本体列表
        """
        ontologies = []
        
        try:
            # 尝试提取JSON部分 - 修复正则表达式匹配多行JSON
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 如果没有markdown格式，尝试寻找JSON对象
                json_match = re.search(r'(\{[^{}]*"ontologies"[^{}]*\[.*?\]\s*\})', llm_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # 最后尝试直接解析
                    json_str = llm_response.strip()
            
            # 解析JSON
            parsed_data = json.loads(json_str)
            
            if "ontologies" in parsed_data and isinstance(parsed_data["ontologies"], list):
                for ont_data in parsed_data["ontologies"]:
                    if isinstance(ont_data, dict) and "name" in ont_data:
                        ontology = {
                            "name": ont_data.get("name", "").strip(),
                            "description": ont_data.get("description", "").strip(),
                            "relationships": ont_data.get("relationships", []),
                            "source_cluster": source_cluster,
                            "raw_response": llm_response
                        }
                        
                        # 验证本体有效性
                        if ontology["name"] and ontology["description"]:
                            ontologies.append(ontology)
                        else:
                            self.logger.warning(f"[解析] 跳过无效本体: {ont_data}")
            
        except json.JSONDecodeError as e:
            self.logger.error(f"[解析] JSON解析失败: {e}")
            self.logger.debug(f"[解析] 尝试的JSON字符串: {json_str[:200]}...")
            self.logger.debug(f"[解析] 原始响应前500字符: {llm_response[:500]}...")
            
            # 回退解析策略
            ontologies = self._fallback_parse_ontologies(llm_response, source_cluster)
        
        except Exception as e:
            self.logger.error(f"[解析] 本体解析失败: {e}")
            
            # 回退解析策略
            ontologies = self._fallback_parse_ontologies(llm_response, source_cluster)
        
        return ontologies
    
    def _fallback_parse_ontologies(self, llm_response: str, source_cluster: int) -> List[Dict[str, Any]]:
        """
        回退解析策略，当JSON解析失败时使用
        
        Args:
            llm_response: LLM响应
            source_cluster: 来源聚类ID
            
        Returns:
            List[Dict]: 解析出的本体列表
        """
        ontologies = []
        
        try:
            # 简单的文本解析
            lines = llm_response.split('\n')
            current_ontology = None
            
            for line in lines:
                line = line.strip()
                
                # 查找本体名称
                if line.startswith('name:') or line.startswith('Name:') or line.startswith('概念:'):
                    if current_ontology and current_ontology.get('name'):
                        ontologies.append(current_ontology)
                    
                    name = re.sub(r'^(name:|Name:|概念:)\s*', '', line, flags=re.IGNORECASE).strip()
                    current_ontology = {
                        "name": name,
                        "description": "",
                        "relationships": [],
                        "source_cluster": source_cluster,
                        "raw_response": llm_response
                    }
                
                # 查找描述
                elif line.startswith('description:') or line.startswith('Description:') or line.startswith('描述:'):
                    if current_ontology:
                        desc = re.sub(r'^(description:|Description:|描述:)\s*', '', line, flags=re.IGNORECASE).strip()
                        current_ontology["description"] = desc
                
                # 查找关系
                elif line.startswith('relationship:') or line.startswith('Relationship:') or line.startswith('关系:'):
                    if current_ontology:
                        rel = re.sub(r'^(relationship:|Relationship:|关系:)\s*', '', line, flags=re.IGNORECASE).strip()
                        if rel:
                            current_ontology["relationships"].append(rel)
            
            # 添加最后一个本体
            if current_ontology and current_ontology.get('name'):
                ontologies.append(current_ontology)
        
        except Exception as e:
            self.logger.error(f"[回退解析] 失败: {e}")
        
        # 如果解析不出来，抛出错误
        if not ontologies:
            raise ValueError(f"聚类 {source_cluster} 无法从LLM响应中解析出有效本体: {llm_response[:200]}...")
        
        return ontologies
    
    def _deduplicate_and_merge_ontologies(self, ontologies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        第二阶段：本体去重和合并（两步法：先名称合并，再语义合并）
        
        Args:
            ontologies: 原始本体列表
            
        Returns:
            List[Dict]: 去重后的本体列表
        """
        if not ontologies:
            return []
        
        self.logger.info(f"[第二阶段] 开始两步法去重处理 {len(ontologies)} 个本体")
        
        # 第一步：按名称合并同名本体
        self.logger.info(f"[第二阶段] 步骤1：按名称合并同名本体")
        name_merged_ontologies = self._merge_by_exact_name(ontologies)
        self.logger.info(f"[第二阶段] 名称合并完成：{len(ontologies)} -> {len(name_merged_ontologies)} 个本体")
        
        # 第二步：对合并后的本体进行语义相似度去重
        if len(name_merged_ontologies) <= 1:
            self.logger.info(f"[第二阶段] 只有 {len(name_merged_ontologies)} 个本体，跳过语义去重")
            return name_merged_ontologies
        
        self.logger.info(f"[第二阶段] 步骤2：对 {len(name_merged_ontologies)} 个本体进行语义相似度去重")
        
        # 计算合并后本体的嵌入向量
        try:
            ontology_names = [ont["name"] for ont in name_merged_ontologies]
            name_embeddings = self._compute_embeddings_concurrent(ontology_names)
            self.logger.info(f"[第二阶段] 计算了 {len(name_embeddings)} 个本体的嵌入向量")
        except Exception as e:
            self.logger.error(f"[第二阶段] 计算嵌入向量失败: {e}")
            raise RuntimeError(f"嵌入向量计算失败: {e}") from e
        
        # 语义相似度聚类
        semantic_groups = self._semantic_similarity_clustering(name_merged_ontologies, name_embeddings)
        
        # 合并语义相似的本体组
        final_ontologies = self._merge_semantic_groups_concurrent(semantic_groups)
        
        self.logger.info(f"[第二阶段] 两步法去重完成：{len(ontologies)} -> {len(final_ontologies)} 个本体")
        
        return final_ontologies
    
    def _merge_by_exact_name(self, ontologies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        按精确名称合并同名本体
        
        Args:
            ontologies: 本体列表
            
        Returns:
            List[Dict]: 合并后的本体列表
        """
        from collections import defaultdict
        
        name_groups = defaultdict(list)
        
        # 按名称分组
        for ont in ontologies:
            normalized_name = ont["name"].lower().strip()
            name_groups[normalized_name].append(ont)
        
        merged_ontologies = []
        duplicate_count = 0
        
        # 分离单个本体和需要合并的组
        single_ontologies = []
        merge_groups = []
        
        for name, group in name_groups.items():
            if len(group) == 1:
                # 单独的本体直接添加
                single_ontologies.extend(group)
            else:
                # 多个同名本体需要合并
                duplicate_count += len(group)
                self.logger.info(f"[名称合并] 发现 {len(group)} 个同名本体: {group[0]['name']}")
                merge_groups.append(group)
        
        # 直接添加单个本体
        merged_ontologies.extend(single_ontologies)
        self.logger.info(f"[名称合并] 直接添加 {len(single_ontologies)} 个单独本体")
        
        # 并发处理需要合并的同名组
        if merge_groups:
            max_workers = min(len(merge_groups), 200)  # 名称合并可以用更多并发
            self.logger.info(f"[名称合并] 开始并发合并 {len(merge_groups)} 个同名组")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有合并任务
                future_to_group = {
                    executor.submit(self._merge_ontology_group, group): group
                    for group in merge_groups
                }
                
                # 收集合并结果
                for future in concurrent.futures.as_completed(future_to_group):
                    group = future_to_group[future]
                    try:
                        merged_ontology = future.result()
                        merged_ontologies.append(merged_ontology)
                        self.logger.info(f"[名称合并] 成功合并 {len(group)} 个同名本体: {merged_ontology['name']}")
                    except Exception as e:
                        self.logger.error(f"[名称合并] 合并本体组失败: {e}")
                        raise RuntimeError(f"同名本体组合并失败: {e}") from e
        
        if duplicate_count > 0:
            self.logger.info(f"[名称合并] 总共合并了 {duplicate_count} 个同名本体")
        
        return merged_ontologies
    
    def _semantic_similarity_clustering(self, ontologies: List[Dict[str, Any]], 
                                       embeddings: np.ndarray) -> List[List[Dict[str, Any]]]:
        """
        基于语义相似度对本体进行聚类
        
        Args:
            ontologies: 本体列表
            embeddings: 对应的嵌入向量
            
        Returns:
            List[List[Dict]]: 聚类结果
        """
        n = len(ontologies)
        visited = [False] * n
        groups = []
        
        self.logger.info(f"[语义聚类] 对 {n} 个本体进行语义相似度聚类")
        
        for i in range(n):
            if visited[i]:
                continue
            
            current_group = [ontologies[i]]
            visited[i] = True
            
            # 计算与所有未访问本体的相似度
            for j in range(i + 1, n):
                if visited[j]:
                    continue
                
                similarity = self._calculate_similarity(embeddings[i], embeddings[j])
                
                if similarity > self.similarity_threshold:
                    current_group.append(ontologies[j])
                    visited[j] = True
                    self.logger.debug(f"[语义聚类] 发现相似本体: {ontologies[i]['name']} <-> {ontologies[j]['name']} (相似度: {similarity:.3f})")
            
            groups.append(current_group)
        
        semantic_duplicates = len([g for g in groups if len(g) > 1])
        total_semantic_duplicates = sum(len(g) for g in groups if len(g) > 1)
        
        self.logger.info(f"[语义聚类] 发现 {semantic_duplicates} 个语义相似组，涉及 {total_semantic_duplicates} 个本体")
        
        return groups
    
    def _merge_semantic_groups_concurrent(self, semantic_groups: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        并发合并语义相似的本体组
        
        Args:
            semantic_groups: 语义聚类结果
            
        Returns:
            List[Dict]: 合并后的本体列表
        """
        final_ontologies = []
        
        # 分离单个本体和需要合并的组
        single_ontologies = []
        merge_groups = []
        
        for group in semantic_groups:
            if len(group) == 1:
                single_ontologies.extend(group)
            else:
                merge_groups.append(group)
        
        # 直接添加单个本体
        final_ontologies.extend(single_ontologies)
        self.logger.info(f"[语义合并] 直接添加 {len(single_ontologies)} 个单独本体")
        
        # 并发处理需要合并的语义相似组
        if merge_groups:
            max_workers = min(len(merge_groups), 200)  # 限制并发数量
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有合并任务
                future_to_group = {
                    executor.submit(self._merge_ontology_group, group): group
                    for group in merge_groups
                }
                
                # 收集合并结果
                for future in concurrent.futures.as_completed(future_to_group):
                    group = future_to_group[future]
                    try:
                        merged_ontology = future.result()
                        final_ontologies.append(merged_ontology)
                        self.logger.info(f"[语义合并] 成功合并 {len(group)} 个语义相似本体: {merged_ontology['name']}")
                    except Exception as e:
                        self.logger.error(f"[语义合并] 合并本体组失败: {e}")
                        raise RuntimeError(f"语义本体组合并失败: {e}") from e
        
        self.logger.info(f"[语义合并] 完成，最终得到 {len(final_ontologies)} 个本体")
        return final_ontologies
    
    def _compute_embeddings_concurrent(self, texts: List[str], batch_size: int = 50) -> np.ndarray:
        """
        并发计算文本嵌入向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            np.ndarray: 嵌入向量数组
        """
        if len(texts) <= batch_size:
            # 小批量直接处理
            return self.embedding_model.encode_batch(texts)
        
        self.logger.info(f"[嵌入计算] 开始并发计算 {len(texts)} 个文本的嵌入向量")
        
        # 动态获取嵌入维度
        embedding_dim = self._get_embedding_dimension()
        
        # 分批处理
        text_batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        all_embeddings = []
        
        # 使用线程池并发处理批次
        max_workers = min(len(text_batches), self.max_concurrent_embedding)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有批次任务
            future_to_batch = {
                executor.submit(self.embedding_model.encode_batch, batch): i
                for i, batch in enumerate(text_batches)
            }
            
            # 收集结果（保持顺序）
            batch_results = [None] * len(text_batches)
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_embeddings = future.result()
                    batch_results[batch_idx] = batch_embeddings
                    self.logger.debug(f"[嵌入计算] 批次 {batch_idx + 1}/{len(text_batches)} 完成")
                except Exception as e:
                    self.logger.error(f"[嵌入计算] 批次 {batch_idx + 1} 失败: {e}")
                    raise RuntimeError(f"嵌入计算批次 {batch_idx + 1} 失败: {e}") from e
        
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
    
    def _get_embedding_dimension(self) -> int:
        """
        动态获取嵌入模型的维度
        
        Returns:
            int: 嵌入向量维度
        """
        try:
            # 使用一个简单的测试文本来获取维度
            test_embedding = self.embedding_model.encode_batch(["test"])
            embedding_dim = test_embedding.shape[1]
            self.logger.debug(f"[嵌入计算] 动态获取嵌入维度: {embedding_dim}")
            return embedding_dim
        except Exception as e:
            self.logger.error(f"[嵌入计算] 无法获取嵌入维度: {e}")
            raise RuntimeError(f"无法获取嵌入模型维度: {e}") from e
    
    def _compute_similarity_matrix_batched(self, embeddings: np.ndarray, batch_size: int = 1000) -> np.ndarray:
        """
        分批计算相似度矩阵以避免内存溢出
        """
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
    
    def _merge_ontology_group(self, ontology_group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        合并一组重复的本体
        
        Args:
            ontology_group: 重复本体组
            
        Returns:
            Dict: 合并后的本体
        """
        if len(ontology_group) == 1:
            return ontology_group[0]
        
        self.logger.info(f"[合并] 正在合并 {len(ontology_group)} 个重复本体")
        
        # 收集所有描述和关系
        descriptions = [ont["description"] for ont in ontology_group if ont["description"]]
        all_relationships = []
        source_clusters = []
        
        for ont in ontology_group:
            all_relationships.extend(ont.get("relationships", []))
            # 安全地获取source_cluster，可能是单个ID或ID列表
            if "source_cluster" in ont:
                source_clusters.append(ont["source_cluster"])
            elif "source_clusters" in ont:
                source_clusters.extend(ont["source_clusters"])
            else:
                # 如果都没有，使用默认值
                source_clusters.append(-1)
        
        # 使用LLM合并描述和关系
        merged_info = self._llm_merge_ontology_info(
            ontology_group[0]["name"],  # 使用第一个本体的名称
            descriptions,
            all_relationships
        )
        
        # 构建合并后的本体
        merged_ontology = {
            "name": ontology_group[0]["name"],  # 保持一致的名称
            "description": merged_info["description"],
            "relationships": merged_info["relationships"],
            "source_clusters": list(set(source_clusters)),  # 记录所有来源聚类
            "merged_from_count": len(ontology_group)
        }
        
        return merged_ontology
    
    def _llm_merge_ontology_info(self, ontology_name: str, descriptions: List[str], 
                                relationships: List[str]) -> Dict[str, Any]:
        """
        使用LLM合并本体的描述和关系（线程安全版本，带重试逻辑）
        
        Args:
            ontology_name: 本体名称
            descriptions: 描述列表
            relationships: 关系列表
            
        Returns:
            Dict: 包含merged_description和merged_relationships
        """
        # 构建合并提示词
        prompt = self._build_merge_prompt(ontology_name, descriptions, relationships)
        
        max_retries = 10  # 合并阶段重试次数较少
        
        for attempt in range(max_retries):
            try:
                response = self.llm_model.generate_single(prompt)
                merged_info = self._parse_merge_response(response)
                
                if merged_info.get("description"):  # 成功解析到描述
                    if attempt > 0:
                        self.logger.info(f"[LLM合并] {ontology_name} 第 {attempt + 1} 次尝试成功")
                    return merged_info
                else:
                    self.logger.warning(f"[LLM合并] {ontology_name} 第 {attempt + 1} 次尝试未解析到有效内容")
                    
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.warning(f"[LLM合并] {ontology_name} 第 {attempt + 1} 次尝试JSON解析失败: {e}")
                if attempt == max_retries - 1:
                    self.logger.error(f"[LLM合并] {ontology_name} 达到最大重试次数，JSON解析持续失败")
                    raise RuntimeError(f"本体合并失败 {ontology_name}: JSON解析持续失败: {e}") from e
                continue
                
            except Exception as e:
                self.logger.error(f"[LLM合并] {ontology_name} 第 {attempt + 1} 次尝试失败: {e}")
                if attempt == max_retries - 1:
                    raise RuntimeError(f"本体合并失败 {ontology_name}: {e}") from e
                continue
        
        # 如果所有重试都失败
        raise RuntimeError(f"本体合并失败 {ontology_name}: 经过 {max_retries} 次重试仍然失败")
    
    def _merge_ontology_groups_concurrent(self, duplicate_groups: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        并发合并多个本体组
        
        Args:
            duplicate_groups: 重复本体组列表
            
        Returns:
            List[Dict]: 合并后的本体列表
        """
        self.logger.info(f"[第二阶段] 开始并发合并 {len(duplicate_groups)} 个本体组")
        
        merged_ontologies = []
        
        # 分离单个本体和需要合并的组
        single_ontologies = []
        merge_groups = []
        
        for group in duplicate_groups:
            if len(group) == 1:
                single_ontologies.extend(group)
            else:
                merge_groups.append(group)
        
        # 直接添加单个本体
        merged_ontologies.extend(single_ontologies)
        self.logger.info(f"[第二阶段] 直接添加 {len(single_ontologies)} 个单独本体")
        
        # 并发处理需要合并的组
        if merge_groups:
            max_workers = min(len(merge_groups), 3)  # 限制并发数量
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有合并任务
                future_to_group = {
                    executor.submit(self._merge_ontology_group, group): group
                    for group in merge_groups
                }
                
                # 收集合并结果
                for future in concurrent.futures.as_completed(future_to_group):
                    group = future_to_group[future]
                    try:
                        merged_ontology = future.result()
                        merged_ontologies.append(merged_ontology)
                        self.logger.info(f"[第二阶段] 成功合并 {len(group)} 个重复本体: {merged_ontology['name']}")
                    except Exception as e:
                        self.logger.error(f"[第二阶段] 合并本体组失败: {e}")
                        raise RuntimeError(f"本体组合并失败: {e}") from e
        
        self.logger.info(f"[第二阶段] 并发合并完成，最终得到 {len(merged_ontologies)} 个本体")
        return merged_ontologies
    
    def _build_merge_prompt(self, ontology_name: str, descriptions: List[str], 
                           relationships: List[str]) -> str:
        """
        构建本体信息合并的提示词
        
        Args:
            ontology_name: 本体名称
            descriptions: 描述列表
            relationships: 关系列表
            
        Returns:
            str: 合并提示词
        """
        descriptions_text = "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(descriptions)])
        relationships_text = "\n".join([f"- {rel}" for rel in relationships])
        
        prompt = f"""Please merge multiple descriptions and relationship information about the ontology "{ontology_name}".

**Task Requirements:**
1. Merge multiple descriptions into one complete, accurate, and concise description
2. Remove duplicate relationships and preserve unique relationship information
3. Ensure merged information is consistent and conflict-free
4. Output in JSON format

**Input Descriptions:**
{descriptions_text}

**Input Relationships:**
{relationships_text}

**Output Format:**
```json
{{
  "description": "Complete merged description",
  "relationships": ["Deduplicated relationship 1", "Deduplicated relationship 2", "..."]
}}
```

Please perform the merge:"""
        
        return prompt
    
    def _parse_merge_response(self, response: str) -> Dict[str, Any]:
        """
        解析LLM合并响应
        
        Args:
            response: LLM响应
            
        Returns:
            Dict: 包含description和relationships
        """
        try:
            # 尝试提取JSON
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response.strip()
            
            parsed = json.loads(json_str)
            
            return {
                "description": parsed.get("description", ""),
                "relationships": parsed.get("relationships", [])
            }
        
        except Exception as e:
            self.logger.error(f"[合并解析] JSON解析失败: {e}")
            raise ValueError(f"合并响应解析失败: {e}") from e
    
    def _convert_to_compatible_format(self, merged_ontologies: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        将合并后的本体转换为与现有系统兼容的格式
        
        Args:
            merged_ontologies: 合并后的本体列表
            
        Returns:
            Dict[int, Dict]: 兼容格式的结果
        """
        self.logger.info(f"[格式转换] 将 {len(merged_ontologies)} 个本体转换为兼容格式")
        
        # 为每个本体分配一个虚拟聚类ID
        compatible_result = {}
        
        for i, ontology in enumerate(merged_ontologies):
            # 使用负数作为虚拟聚类ID，避免与原始聚类ID冲突
            virtual_cluster_id = -(i + 1)
            
            # 构建context：本体名称 + 描述 + 关系
            relationships_text = ""
            if ontology.get("relationships"):
                relationships_text = "\n关系：\n" + "\n".join([f"- {rel}" for rel in ontology["relationships"]])
            
            context = f"{ontology['name']}：{ontology['description']}{relationships_text}"
            
            # 构建兼容的总结结果
            summary = {
                'cluster_id': virtual_cluster_id,
                'document_count': ontology.get("merged_from_count", 1),
                'ontology_summary': context,  # 这是RAG系统会使用的主要内容
                'key_concepts': [ontology['name']],  # 保持向后兼容
                'relationships': ontology.get("relationships", []),
                'source_clusters': ontology.get("source_clusters", []),
                'ontology_name': ontology['name'],  # 新增：本体名称作为标题
                'original_documents': []  # 空列表，因为本体是从多个聚类合并而来
            }
            
            compatible_result[virtual_cluster_id] = summary
        
        self.logger.info(f"[格式转换] 完成转换，生成 {len(compatible_result)} 个兼容条目")
        
        return compatible_result
    
    def _save_ontology_details(self, original_clusters: Dict[int, List[Dict[str, Any]]], 
                              raw_ontologies: List[Dict[str, Any]], 
                              merged_ontologies: List[Dict[str, Any]]):
        """
        保存本体处理详情
        
        Args:
            original_clusters: 原始聚类结果
            raw_ontologies: 原始本体列表
            merged_ontologies: 合并后的本体列表
        """
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
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
                'similarity_threshold': self.similarity_threshold,
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
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"ontology_processing_details_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # 保存到文件
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(processing_details, f, ensure_ascii=False, indent=2)
            self.logger.info(f"[保存详情] 处理详情已保存: {filepath}")
        except Exception as e:
            self.logger.error(f"[保存详情] 保存失败: {e}")
    
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
            'processing_method': 'two_stage_deduplication'
        }
    
    def _generate_cluster_signature(self, clusters: Dict[int, List[Dict[str, Any]]]) -> str:
        """
        为聚类数据生成唯一签名
        
        Args:
            clusters: 聚类数据
            
        Returns:
            str: 唯一签名字符串
        """
        # 生成聚类内容的哈希值
        cluster_content = []
        for cluster_id in sorted(clusters.keys()):
            docs = clusters[cluster_id]
            doc_contents = []
            for doc in docs:
                # 使用关键字段生成签名
                content_str = f"{doc.get('title', '')}{doc.get('context', '')}"
                doc_contents.append(content_str)
            cluster_content.append(f"cluster_{cluster_id}:" + "|".join(sorted(doc_contents)))
        
        signature_str = "||".join(cluster_content)
        signature_hash = hashlib.md5(signature_str.encode('utf-8')).hexdigest()[:12]
        
        return signature_hash
    
    def _extract_ontologies_from_clusters_with_cache(self, clusters: Dict[int, List[Dict[str, Any]]], 
                                                   signature: str) -> List[Dict[str, Any]]:
        """
        带缓存的第一阶段本体抽取
        
        Args:
            clusters: 聚类数据
            signature: 数据签名
            
        Returns:
            List[Dict]: 原始本体列表
        """
        cache_dir = os.path.join(self.output_dir, "cache", "ontologies")
        os.makedirs(cache_dir, exist_ok=True)
        
        stage1_cache_file = os.path.join(cache_dir, f"stage1_raw_ontologies_{signature}.json")
        
        # 尝试从缓存加载
        if os.path.exists(stage1_cache_file):
            try:
                with open(stage1_cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                self.logger.info(f"[缓存] 从缓存加载第一阶段结果: {stage1_cache_file}")
                return cached_data
            except Exception as e:
                self.logger.warning(f"[缓存] 加载第一阶段缓存失败: {e}")
        
        # 缓存不存在或加载失败，执行计算
        self.logger.info("[缓存] 第一阶段缓存不存在，开始计算")
        raw_ontologies = self._extract_ontologies_from_clusters(clusters)
        
        # 保存到缓存
        try:
            # 清理数据以确保JSON序列化兼容性
            cleaned_ontologies = self._clean_data_for_json(raw_ontologies)
            with open(stage1_cache_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned_ontologies, f, ensure_ascii=False, indent=2)
            self.logger.info(f"[缓存] 第一阶段结果已缓存: {stage1_cache_file}")
        except Exception as e:
            self.logger.error(f"[缓存] 保存第一阶段缓存失败: {e}")
        
        return raw_ontologies
    
    def _deduplicate_and_merge_ontologies_with_cache(self, raw_ontologies: List[Dict[str, Any]], 
                                                   signature: str) -> List[Dict[str, Any]]:
        """
        带缓存的第二阶段本体去重和合并
        
        Args:
            raw_ontologies: 原始本体列表
            signature: 数据签名
            
        Returns:
            List[Dict]: 去重后的本体列表
        """
        cache_dir = os.path.join(self.output_dir, "cache", "ontologies")
        os.makedirs(cache_dir, exist_ok=True)
        
        stage2_cache_file = os.path.join(cache_dir, f"stage2_merged_ontologies_{signature}.json")
        
        # 尝试从缓存加载
        if os.path.exists(stage2_cache_file):
            try:
                with open(stage2_cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                self.logger.info(f"[缓存] 从缓存加载第二阶段结果: {stage2_cache_file}")
                return cached_data
            except Exception as e:
                self.logger.warning(f"[缓存] 加载第二阶段缓存失败: {e}")
        
        # 缓存不存在或加载失败，执行计算
        self.logger.info("[缓存] 第二阶段缓存不存在，开始计算")
        merged_ontologies = self._deduplicate_and_merge_ontologies(raw_ontologies)
        
        # 保存到缓存
        try:
            # 清理数据以确保JSON序列化兼容性
            cleaned_ontologies = self._clean_data_for_json(merged_ontologies)
            with open(stage2_cache_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned_ontologies, f, ensure_ascii=False, indent=2)
            self.logger.info(f"[缓存] 第二阶段结果已缓存: {stage2_cache_file}")
        except Exception as e:
            self.logger.error(f"[缓存] 保存第二阶段缓存失败: {e}")
        
        return merged_ontologies
    
    def _clean_data_for_json(self, data):
        """
        清理数据以确保JSON序列化兼容性，将numpy类型转换为Python原生类型
        
        Args:
            data: 要清理的数据
            
        Returns:
            清理后的数据
        """
        import numpy as np
        
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