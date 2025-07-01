#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本体合并模块
负责合并重复的本体
"""

import json
import logging
import concurrent.futures
from typing import List, Dict, Any
from .llm_model import LLMModel
from .prompt_manager import PromptManager
from .retry_utils import retry_with_logging, is_valid_json_response
from .ontology_config import OntologyConfig


class OntologyMerger:
    """本体合并器"""
    
    def __init__(self):
        """初始化本体合并器"""
        self.logger = logging.getLogger(__name__)
        self.llm_model = LLMModel(llm_type="summary")
        self.prompt_manager = PromptManager()
    
    def merge_ontology_groups(self, ontology_groups: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        合并本体组
        
        Args:
            ontology_groups: 本体组列表
            
        Returns:
            List[Dict]: 合并后的本体列表
        """
        if not ontology_groups:
            return []
        
        self.logger.info(f"[本体合并] 开始合并 {len(ontology_groups)} 个本体组")
        
        final_ontologies = []
        
        # 分离单个本体和需要合并的组
        single_ontologies = []
        merge_groups = []
        
        for group in ontology_groups:
            if len(group) == 1:
                single_ontologies.extend(group)
            else:
                merge_groups.append(group)
        
        # 直接添加单个本体
        final_ontologies.extend(single_ontologies)
        self.logger.info(f"[本体合并] 直接添加 {len(single_ontologies)} 个单独本体")
        
        # 处理需要合并的组
        if merge_groups:
            merged_ontologies = self._merge_groups_concurrent(merge_groups)
            final_ontologies.extend(merged_ontologies)
        
        self.logger.info(f"[本体合并] 完成，最终得到 {len(final_ontologies)} 个本体")
        return final_ontologies
    
    def _merge_groups_concurrent(self, merge_groups: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        并发合并多个本体组
        
        Args:
            merge_groups: 需要合并的本体组列表
            
        Returns:
            List[Dict]: 合并后的本体列表
        """
        max_workers = min(len(merge_groups), 200)  # 合并操作可以使用更高并发
        self.logger.info(f"[本体合并] 开始并发合并 {len(merge_groups)} 个本体组 (max_workers={max_workers})")
        
        merged_ontologies = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有合并任务
            future_to_group = {
                executor.submit(self._merge_single_group, group): group
                for group in merge_groups
            }
            
            # 收集合并结果
            for future in concurrent.futures.as_completed(future_to_group):
                group = future_to_group[future]
                try:
                    merged_ontology = future.result()
                    merged_ontologies.append(merged_ontology)
                    self.logger.info(f"[本体合并] 成功合并 {len(group)} 个本体: {merged_ontology['name']}")
                except Exception as e:
                    self.logger.error(f"[本体合并] 合并本体组失败: {e}")
                    raise RuntimeError(f"本体组合并失败: {e}") from e
        
        return merged_ontologies
    
    def _merge_single_group(self, ontology_group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        合并一组重复的本体
        
        Args:
            ontology_group: 重复本体组
            
        Returns:
            Dict: 合并后的本体
        """
        if len(ontology_group) == 1:
            return ontology_group[0]
        
        self.logger.debug(f"[本体合并] 正在合并 {len(ontology_group)} 个重复本体")
        
        # 收集所有描述和关系
        descriptions = [ont["description"] for ont in ontology_group if ont["description"]]
        all_relationships = []
        source_clusters = []
        
        for ont in ontology_group:
            all_relationships.extend(ont.get("relationships", []))
            # 安全地获取source_cluster
            if "source_cluster" in ont:
                source_clusters.append(ont["source_cluster"])
            elif "source_clusters" in ont:
                source_clusters.extend(ont["source_clusters"])
            else:
                source_clusters.append(-1)
        
        # 使用LLM合并描述和关系
        merged_info = self._merge_ontology_info(
            ontology_group[0]["name"],
            descriptions,
            all_relationships
        )
        
        # 构建合并后的本体
        merged_ontology = {
            "name": ontology_group[0]["name"],
            "description": merged_info["description"],
            "relationships": merged_info["relationships"],
            "source_clusters": list(set(source_clusters)),
            "merged_from_count": len(ontology_group)
        }
        
        return merged_ontology
    
    @retry_with_logging(max_retries=OntologyConfig.MAX_MERGE_RETRIES,
                       exception_types=(json.JSONDecodeError, ValueError, RuntimeError))
    def _merge_ontology_info(self, ontology_name: str, descriptions: List[str], 
                           relationships: List[str]) -> Dict[str, Any]:
        """
        使用LLM合并本体的描述和关系
        
        Args:
            ontology_name: 本体名称
            descriptions: 描述列表
            relationships: 关系列表
            
        Returns:
            Dict: 包含merged_description和merged_relationships
        """
        # 构建合并提示词
        prompt = self.prompt_manager.build_merge_prompt(ontology_name, descriptions, relationships)
        
        # 调用LLM
        response = self.llm_model.generate_single(prompt)
        
        # 验证响应质量
        if not is_valid_json_response(response):
            self.logger.warning(f"[LLM合并] {ontology_name} 响应格式无效")
            raise ValueError(f"本体合并响应格式无效: {ontology_name}")
        
        # 解析响应
        merged_info = self._parse_merge_response(response)
        
        if not merged_info.get("description"):
            self.logger.warning(f"[LLM合并] {ontology_name} 未解析到有效内容")
            raise ValueError(f"本体合并未解析到有效内容: {ontology_name}")
        
        return merged_info
    
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
            import re
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
    
    def convert_to_compatible_format(self, merged_ontologies: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        将合并后的本体转换为与现有系统兼容的格式
        
        Args:
            merged_ontologies: 合并后的本体列表
            
        Returns:
            Dict[int, Dict]: 兼容格式的结果
        """
        self.logger.info(f"[格式转换] 将 {len(merged_ontologies)} 个本体转换为兼容格式")
        
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
                'ontology_summary': context,
                'key_concepts': [ontology['name']],
                'relationships': ontology.get("relationships", []),
                'source_clusters': ontology.get("source_clusters", []),
                'ontology_name': ontology['name'],
                'original_documents': []
            }
            
            compatible_result[virtual_cluster_id] = summary
        
        self.logger.info(f"[格式转换] 完成转换，生成 {len(compatible_result)} 个兼容条目")
        
        return compatible_result