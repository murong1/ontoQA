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
from tqdm import tqdm
from .llm_model import LLMModel
from .prompt_manager import PromptManager
from .retry_utils import retry_with_logging
from config import Config


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
        
        # 使用tqdm显示进度条
        for cluster_info in tqdm(cluster_data, desc="本体抽取进度", unit="个聚类"):
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
        max_workers = min(len(cluster_data), Config.MAX_CONCURRENT_LLM)
        self.logger.info(f"[本体抽取] 大规模数据，使用并发处理 (max_workers={max_workers})")
        
        all_ontologies = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_cluster = {
                executor.submit(self._extract_from_single_cluster, cluster_info): cluster_info
                for cluster_info in cluster_data
            }
            
            # 使用tqdm显示进度条
            with tqdm(total=len(future_to_cluster), desc="本体抽取进度", unit="个聚类") as pbar:
                # 收集结果
                for future in concurrent.futures.as_completed(future_to_cluster):
                    cluster_info = future_to_cluster[future]
                    cluster_id = cluster_info['cluster_id']
                    
                    try:
                        ontologies = future.result()
                        all_ontologies.extend(ontologies)
                        self.logger.debug(f"[本体抽取] 聚类 {cluster_id} 抽取到 {len(ontologies)} 个本体")
                        pbar.set_postfix({"已完成": f"聚类{cluster_id}", "本体数": len(ontologies)})
                    except Exception as e:
                        self.logger.error(f"[本体抽取] 聚类 {cluster_id} 处理失败: {e}")
                        pbar.set_postfix({"失败": f"聚类{cluster_id}"})
                        raise RuntimeError(f"聚类 {cluster_id} 本体抽取失败: {e}") from e
                    finally:
                        pbar.update(1)
        
        self.logger.info(f"[本体抽取] 并发处理完成，总共抽取到 {len(all_ontologies)} 个原始本体")
        return all_ontologies
    
    @retry_with_logging(max_retries=Config.MAX_EXTRACTION_RETRIES,
                       exception_types=(json.JSONDecodeError, ValueError, RuntimeError))
    def _extract_from_single_cluster(self, cluster_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        处理单个聚类的本体抽取，使用统一的验证和解析流程
        
        Args:
            cluster_info: 聚类信息
            
        Returns:
            List[Dict]: 从该聚类提取的本体列表
        """
        cluster_id = cluster_info['cluster_id']
        prompt = cluster_info['prompt']
        
        # 调用LLM进行本体抽取
        llm_response = self.llm_model.generate_single(prompt)
        
        # 使用统一验证和解析方法，任何验证失败都会抛出异常触发重试
        ontologies = self._extract_validate_and_parse_ontologies(llm_response, cluster_id)
        
        self.logger.debug(f"[本体抽取] 聚类 {cluster_id} 成功抽取 {len(ontologies)} 个本体")
        return ontologies
    
    def _extract_validate_and_parse_ontologies(self, llm_response: str, source_cluster: int) -> List[Dict[str, Any]]:
        """
        从LLM响应中提取、验证并解析本体，任何步骤失败都抛出异常
        
        Args:
            llm_response: LLM的响应文本
            source_cluster: 来源聚类ID
            
        Returns:
            List[Dict]: 完全验证和解析的本体列表
            
        Raises:
            ValueError: 任何验证或解析步骤失败
        """
        try:
            # 步骤1: 提取JSON部分
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
            
            # 步骤2: 解析JSON
            parsed_data = json.loads(json_str)
            
            # 步骤3: 验证JSON基本结构
            if not isinstance(parsed_data, dict):
                raise ValueError(f"聚类 {source_cluster} JSON根节点必须是对象")
            
            if "ontologies" not in parsed_data:
                raise ValueError(f"聚类 {source_cluster} JSON缺少'ontologies'字段")
            
            if not isinstance(parsed_data["ontologies"], list):
                raise ValueError(f"聚类 {source_cluster} 'ontologies'字段必须是数组")
            
            if not parsed_data["ontologies"]:
                raise ValueError(f"聚类 {source_cluster} 'ontologies'数组不能为空")
            
            # 步骤4: 验证和解析每个本体对象
            ontologies = []
            for i, ont_data in enumerate(parsed_data["ontologies"]):
                if not isinstance(ont_data, dict):
                    raise ValueError(f"聚类 {source_cluster} 本体 {i} 必须是对象")
                
                # 验证必要字段
                name = ont_data.get("name", "").strip()
                description = ont_data.get("description", "").strip()
                
                if not name:
                    raise ValueError(f"聚类 {source_cluster} 本体 {i} 缺少有效的'name'字段")
                
                if not description:
                    raise ValueError(f"聚类 {source_cluster} 本体 {i} 缺少有效的'description'字段")
                
                # 步骤5: 严格验证关系数据
                relationships = ont_data.get("relationships", [])
                validated_relationships = self._validate_relationships_strict(relationships, source_cluster, i)
                
                # 构建本体对象
                ontology = {
                    "name": name,
                    "description": description,
                    "relationships": validated_relationships,
                    "source_cluster": source_cluster,
                    "raw_response": llm_response
                }
                
                ontologies.append(ontology)
            
            self.logger.debug(f"[本体验证] 聚类 {source_cluster} 成功验证并解析 {len(ontologies)} 个本体")
            return ontologies
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"[本体验证] 聚类 {source_cluster} JSON解析失败: {e}")
            raise ValueError(f"聚类 {source_cluster} JSON格式无效: {e}")
        except Exception as e:
            if isinstance(e, ValueError):
                # 重新抛出已知的验证错误
                raise
            else:
                # 包装未知错误
                self.logger.warning(f"[本体验证] 聚类 {source_cluster} 验证失败: {e}")
                raise ValueError(f"聚类 {source_cluster} 本体验证失败: {e}")
    
    
    def _validate_relationships_strict(self, relationships: List, source_cluster: int, ontology_index: int) -> List[Dict[str, str]]:
        """
        严格验证关系数据格式，只支持结构化格式，relation字段必须符合Subject->Predicate->Object格式
        
        Args:
            relationships: 关系数据列表
            source_cluster: 来源聚类ID
            ontology_index: 本体在数组中的索引
            
        Returns:
            List[Dict]: 验证后的关系列表
            
        Raises:
            ValueError: 关系格式验证失败
        """
        if not isinstance(relationships, list):
            raise ValueError(f"聚类 {source_cluster} 本体 {ontology_index} relationships必须是数组")
        
        validated_relationships = []
        
        for rel_index, rel in enumerate(relationships):
            # 只支持结构化格式: {"relation": "Subject -> Predicate -> Object", "type": "TYPE|OTHER"}
            if not isinstance(rel, dict):
                raise ValueError(f"聚类 {source_cluster} 本体 {ontology_index} 关系 {rel_index} 必须是对象，得到: {type(rel)}")
            
            validated_rel = self._validate_structured_relation(rel, source_cluster, ontology_index, rel_index)
            validated_relationships.append(validated_rel)
        
        return validated_relationships
    
    def _validate_structured_relation(self, rel: Dict, source_cluster: int, ontology_index: int, rel_index: int) -> Dict[str, str]:
        """
        验证结构化关系格式 {"relation": "Subject -> Predicate -> Object", "type": "TYPE|OTHER"}
        """
        if "relation" not in rel:
            raise ValueError(f"聚类 {source_cluster} 本体 {ontology_index} 关系 {rel_index} 缺少'relation'字段")
        
        if "type" not in rel:
            raise ValueError(f"聚类 {source_cluster} 本体 {ontology_index} 关系 {rel_index} 缺少'type'字段")
        
        relation_text = rel["relation"].strip() if isinstance(rel["relation"], str) else ""
        relation_type = rel["type"].strip().upper() if isinstance(rel["type"], str) else ""
        
        if not relation_text:
            raise ValueError(f"聚类 {source_cluster} 本体 {ontology_index} 关系 {rel_index} 'relation'字段不能为空")
        
        if not relation_type:
            raise ValueError(f"聚类 {source_cluster} 本体 {ontology_index} 关系 {rel_index} 'type'字段不能为空")
        
        # 确保类型只能是TYPE或OTHER
        if relation_type not in ["TYPE", "OTHER"]:
            raise ValueError(f"聚类 {source_cluster} 本体 {ontology_index} 关系 {rel_index} 'type'必须是'TYPE'或'OTHER'，得到: {relation_type}")
        
        # 验证relation字段必须符合 "Subject -> Predicate -> Object" 格式
        self._validate_relation_format(relation_text, source_cluster, ontology_index, rel_index)
        
        return {
            "relation": relation_text,
            "type": relation_type
        }
    
    def _validate_relation_format(self, relation_text: str, source_cluster: int, ontology_index: int, rel_index: int):
        """
        验证关系字符串必须符合 "Subject -> Predicate -> Object" 格式
        """
        # 验证是否符合 "Subject -> Predicate -> Object" 格式
        parts = re.split(r'\s*->\s*', relation_text)
        if len(parts) != 3:
            raise ValueError(f"聚类 {source_cluster} 本体 {ontology_index} 关系 {rel_index} "
                           f"relation字段必须符合'Subject -> Predicate -> Object'格式，得到: {relation_text}")
        
        subject, predicate, obj = [part.strip() for part in parts]
        
        # 验证每个部分都不为空
        if not subject:
            raise ValueError(f"聚类 {source_cluster} 本体 {ontology_index} 关系 {rel_index} Subject不能为空")
        
        if not predicate:
            raise ValueError(f"聚类 {source_cluster} 本体 {ontology_index} 关系 {rel_index} Predicate不能为空")
        
        if not obj:
            raise ValueError(f"聚类 {source_cluster} 本体 {ontology_index} 关系 {rel_index} Object不能为空")
        
        # 验证不能有重复的Subject、Predicate、Object
        if subject == predicate or predicate == obj or subject == obj:
            raise ValueError(f"聚类 {source_cluster} 本体 {ontology_index} 关系 {rel_index} "
                           f"Subject、Predicate、Object必须各不相同: {relation_text}")
    
