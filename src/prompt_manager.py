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
3. Extract relationships between ontologies, **especially focus on TYPE relationships** (use unified "is type of" format)
4. Distinguish between two relationship types: TYPE and OTHER
5. Strictly follow JSON format for output

Relationship Types:
- **TYPE**: Hierarchical classification relationships, use unified "is type of" format (e.g., "Dog -> is type of -> Animal")
- **OTHER**: non-is-a semantic relationships between types (e.g., "Algorithm -> used in -> Machine Learning")

**IMPORTANT RULES FOR RELATIONSHIPS**:
1. **BE SPECIFIC**: Every relationship must have a clear, precise semantic meaning
2. **FORBIDDEN PREDICATES**: DO NOT use vague predicates like:
   - "related to", "relates to", "associated with", "connected to", "linked to"
   - These are TOO GENERAL and provide no semantic value
3. **REQUIRED**: Use specific predicates that clearly express the nature of the relationship:
   - Action-based: "produces", "consumes", "transforms", "generates", "requires"
   - Structural: "contains", "consists of", "includes", "part of"
   - Temporal: "precedes", "follows", "occurs during"
   - Spatial: "located in", "adjacent to", "surrounds"
   - Functional: "used for", "enables", "supports", "implements"
   - Causal: "causes", "results in", "depends on", "influences"
   - Ownership: "belongs to", "owned by", "manages", "controls"

Output format example:
```json
{{
  "ontologies": [
    {{
      "name": "Machine Learning",
      "description": "An artificial intelligence technology that enables computer systems to automatically learn and improve through data training algorithms.",
      "relationships": [
        {{
          "relation": "Machine Learning -> is type of -> Artificial Intelligence",
          "type": "TYPE"
        }},
        {{
          "relation": "Machine Learning -> contains -> Supervised Learning",
          "type": "OTHER"
        }},
        {{
          "relation": "Machine Learning -> contains -> Unsupervised Learning",
          "type": "OTHER"
        }},
        {{
          "relation": "Machine Learning -> requires -> Training Data",
          "type": "OTHER"
        }}
      ]
    }},
    {{
      "name": "Supervised Learning",
      "description": "A branch of machine learning that uses labeled training data to learn mappings from inputs to outputs.",
      "relationships": [
        {{
          "relation": "Supervised Learning -> is type of -> Machine Learning",
          "type": "TYPE"
        }},
        {{
          "relation": "Supervised Learning -> requires -> Labeled Data",
          "type": "OTHER"
        }},
        {{
          "relation": "Supervised Learning -> includes -> Classification",
          "type": "OTHER"
        }},
        {{
          "relation": "Supervised Learning -> includes -> Regression",
          "type": "OTHER"
        }}
      ]
    }}
  ]
}}
```

Notes:
- Ontology names should be concise and clear
- Descriptions should be self-contained, not dependent on other ontology definitions
- **Prioritize extracting TYPE relationships** as they are crucial for ontological hierarchies
- Use structured relationship format with explicit type classification
- Relationships can include ontology names or other relevant concepts
- **CRITICAL**: Each relationship predicate must be semantically precise and meaningful
- **NEVER** use generic predicates like "related to" - always specify HOW things are related
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
        
        prompt = f"""Merge multiple descriptions and relationship information for the ontology "{ontology_name}" into a coherent, comprehensive definition.

Requirements:
1. **Description Merging**: Combine multiple descriptions into one complete, accurate, and self-contained definition
2. **Relationship Deduplication**: Remove duplicate relationships while preserving all unique semantic information
3. **Type Classification**: Maintain distinction between TYPE relationships (use unified "is type of" format) and OTHER relationships
4. **Consistency Check**: Ensure merged information is logically consistent and conflict-free
5. **Completeness**: Preserve all important semantic information from input sources
6. **Stricty follow JSON format** for output

Merging Strategy:
- **For Descriptions**: Extract core concepts, combine complementary information, resolve conflicts by choosing more specific/accurate descriptions
- **For Relationships**: Group by relationship type, eliminate exact duplicates, preserve semantically distinct relationships
- **Quality Control**: Ensure the merged ontology maintains conceptual clarity and independence

Input Descriptions:
{descriptions_text}

Input Relationships:
{relationships_text}

Output Format Example:
```json
{{
  "description": "A comprehensive, self-contained description that combines all input descriptions while maintaining clarity and accuracy",
  "relationships": [
    {{
      "relation": "Supervised Learning -> is type of -> Machine Learning",
      "type": "TYPE"
    }},
    {{
      "relation": "Supervised Learning -> contains -> Classification",
      "type": "OTHER"
    }}
  ]
}}
```

Notes:
- Merged description should be concise yet complete, not dependent on other definitions
- **Prioritize preserving TYPE relationships** as they are crucial for ontological hierarchies
- Use structured relationship format consistent with extraction phase
- Ensure JSON format is correct and parseable
- Resolve any semantic conflicts by choosing more accurate or specific information

Please perform the merge:"""
        
        return prompt