#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本体总结模块
负责使用LLM对聚类结果进行本体总结
"""

import logging
from typing import List, Dict, Any
from .llm_model import LLMModel


class OntologySummarizer:
    """本体总结器"""
    
    def __init__(self):
        """
        初始化总结器
        """
        self.logger = logging.getLogger(__name__)
        self.llm_model = LLMModel(llm_type="summary")
    
    def summarize_clusters(self, clusters: Dict[int, List[Dict[str, Any]]]) -> Dict[int, Dict[str, Any]]:
        """
        对聚类结果进行本体总结
        
        Args:
            clusters: 聚类结果
            
        Returns:
            Dict[int, Dict]: 总结结果，包含本体信息
        """
        self.logger.info(f"Starting ontology summarization for {len(clusters)} clusters")
        
        # 准备批量处理数据
        cluster_data = []
        prompts = []
        
        for cluster_id, documents in clusters.items():
            self.logger.info(f"Preparing cluster {cluster_id} with {len(documents)} documents")
            
            # 提取文档内容
            contexts = []
            titles = []
            for doc in documents:
                contexts.append(doc.get('context', ''))
                titles.append(doc.get('title', ''))
            
            # 构建提示词
            prompt = self._build_summarization_prompt(contexts, titles)
            
            cluster_data.append({
                'cluster_id': cluster_id,
                'documents': documents,
                'prompt': prompt
            })
            prompts.append(prompt)
        
        # 使用批量处理
        self.logger.info("Starting batch LLM processing for all clusters")
        try:
            ontology_summaries = self.llm_model.generate_batch(prompts)
        except Exception as e:
            self.logger.error(f"Batch LLM processing failed: {e}")
            # 回退到单个处理
            return self._fallback_to_single_processing(clusters)
        
        # 构建最终结果
        summaries = {}
        for i, cluster_info in enumerate(cluster_data):
            cluster_id = cluster_info['cluster_id']
            documents = cluster_info['documents']
            prompt = cluster_info['prompt']
            ontology_summary = ontology_summaries[i]
            
            summary = {
                'cluster_id': cluster_id,
                'document_count': len(documents),
                'ontology_summary': ontology_summary,
                'key_concepts': self._extract_concepts(ontology_summary),
                'relationships': self._extract_relationships(ontology_summary),
                'prompt_used': prompt,
                'original_documents': documents
            }
            summaries[cluster_id] = summary
        
        self.logger.info(f"Completed ontology summarization for {len(summaries)} clusters")
        return summaries
    
    def _fallback_to_single_processing(self, clusters: Dict[int, List[Dict[str, Any]]]) -> Dict[int, Dict[str, Any]]:
        """
        回退到单个处理模式
        
        Args:
            clusters: 聚类结果
            
        Returns:
            Dict[int, Dict]: 总结结果
        """
        self.logger.warning("Falling back to single processing mode")
        summaries = {}
        for cluster_id, documents in clusters.items():
            self.logger.info(f"Summarizing cluster {cluster_id} with {len(documents)} documents (single mode)")
            summary = self._summarize_single_cluster(cluster_id, documents)
            summaries[cluster_id] = summary
        
        return summaries
    
    def _summarize_single_cluster(self, cluster_id: int, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        对单个聚类进行总结
        
        Args:
            cluster_id: 聚类ID
            documents: 聚类中的文档列表
            
        Returns:
            Dict: 总结结果
        """
        # 提取文档内容
        contexts = []
        titles = []
        for doc in documents:
            contexts.append(doc.get('context', ''))
            titles.append(doc.get('title', ''))
        
        # 构建提示词
        prompt = self._build_summarization_prompt(contexts, titles)
        
        # 调用LLM进行总结
        try:
            ontology_summary = self.llm_model.generate_single(prompt)
        except Exception as e:
            self.logger.error(f"LLM call failed for cluster {cluster_id}: {e}")
            ontology_summary = f"Error generating summary: {str(e)}"
        
        summary = {
            'cluster_id': cluster_id,
            'document_count': len(documents),
            'ontology_summary': ontology_summary,
            'key_concepts': self._extract_concepts(ontology_summary),
            'relationships': self._extract_relationships(ontology_summary),
            'prompt_used': prompt,
            'original_documents': documents
        }
        
        return summary
    
    def _build_summarization_prompt(self, contexts: List[str], titles: List[str]) -> str:
        """
        构建总结提示词
        
        Args:
            contexts: 文档内容列表
            titles: 文档标题列表
            
        Returns:
            str: 构建的提示词
        """
        documents_text = "\n\n".join([
            f"Title: {title}\nContent: {context}" 
            for title, context in zip(titles, contexts)
        ])
        
        prompt = f"""From the following document cluster, please perform the following tasks:
1. Identify the few most important **main ontological concepts**.
2. For each main concept, provide a brief textual description.
3. Extract relationships between concepts. These relationships can involve both main concepts and other less central concepts mentioned in the text.

The desired output format is as follows. Please adhere to it strictly.

Example:

Main Concept: Machine Learning
Description: A field of artificial intelligence that uses statistical techniques to give computer systems the ability to "learn" from data.

Main Concept: Supervised Learning
Description: A subcategory of machine learning and artificial intelligence. It is defined by its use of labeled datasets to train algorithms that to classify data or predict outcomes accurately.

Relationship: Supervised Learning -> is a type of -> Machine Learning
Relationship: Model -> is trained on -> Dataset

Note: In the relationships, concepts like "Model" or "Dataset" might not be main concepts and do not require a separate description. Only provide descriptions for the main concepts.

Documents:
{documents_text}

Please identify the main concepts with their descriptions, and the relationships based on the provided text."""
        
        return prompt
    
    def _extract_concepts(self, summary_text: str) -> List[str]:
        """
        从总结中提取关键概念
        
        Args:
            summary_text: 总结文本
            
        Returns:
            List[str]: 关键概念列表
        """
        # 简单的关键词提取逻辑
        concepts = []
        lines = summary_text.split('\n')
        for line in lines:
            if 'concept' in line.lower() or 'entity' in line.lower():
                # 提取包含概念的行
                concepts.append(line.strip())
        
        # 如果没有找到，返回默认值
        if not concepts:
            concepts = ["general_concept_1", "general_concept_2", "general_concept_3"]
        
        return concepts[:5]  # 最多返回5个概念
    
    def _extract_relationships(self, summary_text: str) -> List[str]:
        """
        从总结中提取关系
        
        Args:
            summary_text: 总结文本
            
        Returns:
            List[str]: 关系列表
        """
        # 简单的关系提取逻辑
        relationships = []
        lines = summary_text.split('\n')
        for line in lines:
            if 'relationship' in line.lower() or 'relation' in line.lower() or 'connect' in line.lower():
                relationships.append(line.strip())
        
        # 如果没有找到，返回默认值
        if not relationships:
            relationships = ["relationship_1", "relationship_2"]
        
        return relationships[:3]  # 最多返回3个关系