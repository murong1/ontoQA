#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提示词管理模块
"""

from typing import List


class PromptManager:
    """提示词管理器"""
    
    @staticmethod
    def build_ontology_extraction_prompt(contexts: List[str], titles: List[str]) -> str:
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
    
    @staticmethod
    def build_merge_prompt(ontology_name: str, descriptions: List[str], 
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