#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本体抽取模块
负责从文档聚类中抽取结构化本体
"""

import json
import logging
import re
import concurrent.futures
from typing import List, Dict, Any
from .llm_model import LLMModel
from .prompt_manager import PromptManager
from .retry_utils import retry_with_logging, is_valid_json_response
from .ontology_config import OntologyConfig


class OntologyExtractor:
    """本体抽取器"""
    
    def __init__(self):
        """初始化本体抽取器"""
        self.logger = logging.getLogger(__name__)
        self.llm_model = LLMModel(llm_type="summary")
        self.prompt_manager = PromptManager()
    
    def extract_from_clusters(self, clusters: Dict[int, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        从聚类中抽取结构化本体
        
        Args:
            clusters: 聚类结果
            
        Returns:
            List[Dict]: 原始本体列表
        """
        if not clusters:
            raise ValueError("聚类数据为空，无法进行本体抽取")
        
        if not all(isinstance(docs, list) and docs for docs in clusters.values()):
            raise ValueError("聚类数据包含空的文档列表")
        
        self.logger.info(f"[本体抽取] 开始处理 {len(clusters)} 个聚类")
        
        # 准备并发处理数据
        cluster_data = self._prepare_cluster_data(clusters)
        
        # 根据数据量选择处理策略
        if len(cluster_data) <= 2:
            return self._process_sequentially(cluster_data)
        else:
            return self._process_concurrently(cluster_data)
    
    def _prepare_cluster_data(self, clusters: Dict[int, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        准备聚类数据以供处理
        
        Args:
            clusters: 聚类结果
            
        Returns:
            List[Dict]: 准备好的聚类数据
        """
        cluster_data = []
        for cluster_id, documents in clusters.items():
            self.logger.debug(f"[本体抽取] 准备聚类 {cluster_id}，包含 {len(documents)} 个文档")
            
            # 提取文档内容
            contexts = [doc.get('context', '') for doc in documents]
            titles = [doc.get('title', '') for doc in documents]
            
            # 构建提示词
            prompt = self.prompt_manager.build_ontology_extraction_prompt(contexts, titles)
            
            cluster_data.append({
                'cluster_id': cluster_id,
                'documents': documents,
                'prompt': prompt
            })
        
        return cluster_data
    
    def _process_sequentially(self, cluster_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        顺序处理聚类
        
        Args:
            cluster_data: 聚类数据
            
        Returns:
            List[Dict]: 抽取的本体列表
        """
        self.logger.info(f"[本体抽取] 小规模数据，使用顺序处理")
        all_ontologies = []
        
        for cluster_info in cluster_data:
            ontologies = self._extract_from_single_cluster(cluster_info)
            all_ontologies.extend(ontologies)
            cluster_id = cluster_info['cluster_id']
            self.logger.debug(f"[本体抽取] 聚类 {cluster_id} 抽取到 {len(ontologies)} 个本体")
        
        return all_ontologies
    
    def _process_concurrently(self, cluster_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        并发处理聚类
        
        Args:
            cluster_data: 聚类数据
            
        Returns:
            List[Dict]: 抽取的本体列表
        """
        max_workers = min(len(cluster_data), OntologyConfig.MAX_CONCURRENT_LLM)
        self.logger.info(f"[本体抽取] 大规模数据，使用并发处理 (max_workers={max_workers})")
        
        all_ontologies = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_cluster = {
                executor.submit(self._extract_from_single_cluster, cluster_info): cluster_info
                for cluster_info in cluster_data
            }
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_cluster):
                cluster_info = future_to_cluster[future]
                cluster_id = cluster_info['cluster_id']
                
                try:
                    ontologies = future.result()
                    all_ontologies.extend(ontologies)
                    self.logger.debug(f"[本体抽取] 聚类 {cluster_id} 抽取到 {len(ontologies)} 个本体")
                except Exception as e:
                    self.logger.error(f"[本体抽取] 聚类 {cluster_id} 处理失败: {e}")
                    raise RuntimeError(f"聚类 {cluster_id} 本体抽取失败: {e}") from e
        
        self.logger.info(f"[本体抽取] 并发处理完成，总共抽取到 {len(all_ontologies)} 个原始本体")
        return all_ontologies
    
    @retry_with_logging(max_retries=OntologyConfig.MAX_EXTRACTION_RETRIES,
                       exception_types=(json.JSONDecodeError, ValueError, RuntimeError))
    def _extract_from_single_cluster(self, cluster_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        处理单个聚类的本体抽取
        
        Args:
            cluster_info: 聚类信息
            
        Returns:
            List[Dict]: 从该聚类提取的本体列表
        """
        cluster_id = cluster_info['cluster_id']
        prompt = cluster_info['prompt']
        
        # 调用LLM进行本体抽取
        llm_response = self.llm_model.generate_single(prompt)
        
        # 验证响应质量
        if not is_valid_json_response(llm_response):
            self.logger.warning(f"[本体抽取] 聚类 {cluster_id} 响应格式无效")
            raise ValueError(f"聚类 {cluster_id} LLM响应格式无效")
        
        # 解析响应
        ontologies = self._parse_ontologies_from_response(llm_response, cluster_id)
        
        if not ontologies:
            self.logger.warning(f"[本体抽取] 聚类 {cluster_id} 未解析到本体")
            raise ValueError(f"聚类 {cluster_id} 未解析到有效本体")
        
        return ontologies
    
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
            # 尝试提取JSON部分
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 如果没有markdown格式，尝试寻找JSON对象
                json_match = re.search(r'(\{[^{}]*"ontologies"[^{}]*\[.*?\]\s*\})', llm_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
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
            self.logger.debug(f"[解析] 原始响应前500字符: {llm_response[:500]}...")
            
            # 使用回退解析策略
            ontologies = self._fallback_parse_ontologies(llm_response, source_cluster)
        
        except Exception as e:
            self.logger.error(f"[解析] 本体解析失败: {e}")
            # 使用回退解析策略
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
            raise ValueError(f"聚类 {source_cluster} 无法从LLM响应中解析出有效本体")
        
        return ontologies