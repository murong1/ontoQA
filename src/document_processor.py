#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档处理模块
负责准备用于索引的文档数据
"""

from typing import List, Dict, Any


class DocumentProcessor:
    """文档处理器"""
    
    @staticmethod
    def prepare_full_index_documents(corpus: List[Dict[str, Any]], 
                                   summaries: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """准备完整索引文档（包含原始文档和总结）"""
        index_documents = []
        
        # 原始文档
        for doc in corpus:
            index_documents.append({
                'doc_type': 'original_document',
                'content': doc.get('context', ''),
                'title': doc.get('title', ''),
                'doc_id': doc.get('id', ''),
                'source': doc
            })
        
        # 总结文档
        for cluster_id, summary in summaries.items():
            index_documents.append({
                'doc_type': 'ontology_summary',
                'content': summary.get('ontology_summary', ''),
                'title': f"Cluster {cluster_id} Summary",
                'cluster_id': cluster_id,
                'doc_id': f"summary_{cluster_id}",
                'source': summary
            })
        
        return index_documents
    
    @staticmethod
    def prepare_documents_only_index(corpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """准备仅包含原始文档的索引"""
        return [{
            'doc_type': 'original_document',
            'content': doc.get('context', ''),
            'title': doc.get('title', ''),
            'doc_id': doc.get('id', ''),
            'source': doc
        } for doc in corpus]
    
    @staticmethod
    def format_context_for_prompt(retrieved_docs: List[Dict[str, Any]]) -> str:
        """为提示词格式化检索到的文档上下文"""
        context_parts = []
        for doc in retrieved_docs:
            doc_type = doc.get('doc_type', 'unknown')
            content = doc.get('content', '')
            title = doc.get('title', '')
            score = doc.get('similarity_score', 0.0)
            
            context_parts.append(f"[{doc_type.upper()}] {title}\n{content}\n(Relevance: {score:.3f})")
        
        return "\n\n".join(context_parts)