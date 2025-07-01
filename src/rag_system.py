#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG系统模块
负责构建检索增强生成系统
"""

import logging
from typing import List, Dict, Any
import json
import os
from datetime import datetime
from .embedding_model import EmbeddingModel
from .llm_model import LLMModel
from .index_cache import IndexCache
from .document_processor import DocumentProcessor
from .faiss_builder import FaissIndexBuilder
from config import Config


class RAGSystem:
    """RAG系统"""
    
    def __init__(self, top_k: int = None, index_cache_dir: str = "cache/indexes", output_dir: str = "results"):
        """初始化RAG系统"""
        self.top_k = top_k if top_k is not None else Config.DEFAULT_TOP_K
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.embedding_model = EmbeddingModel()
        self.llm_model = LLMModel(llm_type="question")
        self.cache = IndexCache(index_cache_dir)
        self.faiss_builder = FaissIndexBuilder()
        
        # 索引相关
        self.faiss_index = None
        self.documents = []
        self.summaries = {}
        self.index_documents = []
        
        # 检索追踪
        self.retrieval_log = []
    
    def _load_from_cache_if_exists(self, cache_key: str) -> bool:
        """如果缓存存在则加载"""
        if self.cache.exists(cache_key):
            self.logger.info("[RAG系统] 找到缓存索引，正在加载...")
            data = self.cache.load(cache_key)
            self.faiss_index = data['faiss_index']
            self.index_documents = data['index_documents']
            self.documents = data['documents']
            self.summaries = data['summaries']
            # 不从缓存加载top_k，保持当前设置
            self.logger.info("[RAG系统] 缓存索引加载完成")
            return True
        
        self.logger.info("[RAG系统] 未找到缓存索引，开始构建新索引...")
        return False
    
    def build_index(self, corpus: List[Dict[str, Any]], summaries: Dict[int, Dict[str, Any]], 
                   corpus_path: str, n_clusters: int):
        """构建包含原始文档和总结的完整索引"""
        cache_key = self.cache.generate_cache_key(corpus_path, n_clusters, "full")
        
        if self._load_from_cache_if_exists(cache_key):
            return
        
        # 构建新索引
        self._store_data(corpus, summaries)
        index_documents = DocumentProcessor.prepare_full_index_documents(corpus, summaries)
        self._build_and_save_index(index_documents, cache_key)
        
        self.logger.info(f"[RAG系统] 完整索引构建完成，包含 {len(index_documents)} 个文档 ({len(corpus)} 原始 + {len(summaries)} 本体)")
    
    def build_index_documents_only(self, corpus: List[Dict[str, Any]], corpus_path: str):
        """构建仅包含原始文档的索引（消融实验用）"""
        cache_key = self.cache.generate_cache_key(corpus_path, 0, "documents_only")
        
        if self._load_from_cache_if_exists(cache_key):
            return
        
        # 构建新索引
        self._store_data(corpus, {})
        index_documents = DocumentProcessor.prepare_documents_only_index(corpus)
        self._build_and_save_index(index_documents, cache_key)
        
        self.logger.info(f"[RAG系统] 文档索引构建完成，包含 {len(index_documents)} 个原始文档（消融实验模式）")
    
    def _save_to_cache(self, cache_key: str):
        """保存索引到缓存"""
        if self.faiss_index is None:
            raise ValueError("No index to save. Please build index first.")
        
        self.cache.save(cache_key, self.faiss_index, self.index_documents, 
                       self.documents, self.summaries, self.top_k)
    
    def _store_data(self, corpus: List[Dict[str, Any]], summaries: Dict[int, Dict[str, Any]]):
        """存储原始数据"""
        self.documents = corpus.copy()
        self.summaries = summaries
    
    def _build_and_save_index(self, index_documents: List[Dict[str, Any]], cache_key: str):
        """构建并保存索引"""
        texts = [doc['content'] for doc in index_documents]
        embeddings = self.embedding_model.encode(texts)
        
        self.index_documents = index_documents
        self.faiss_index = self.faiss_builder.build_index(embeddings)
        self._save_to_cache(cache_key)
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """检索相关文档"""
        if self.faiss_index is None:
            raise ValueError("FAISS index has not been built yet")
        
        query_embedding = self.embedding_model.encode([query])
        scores, indices = self.faiss_builder.search(self.faiss_index, query_embedding, self.top_k)
        
        # 构建检索结果
        retrieved_docs = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                doc = {
                    **self.index_documents[idx],
                    'similarity_score': float(score)
                }
                retrieved_docs.append(doc)
        
        # 记录检索详情
        self._log_retrieval_details(query, retrieved_docs)
        
        return retrieved_docs
    
    def _create_qa_prompt(self, question: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """创建问答提示词"""
        context = DocumentProcessor.format_context_for_prompt(retrieved_docs)
        
        return f"""Based on the following retrieved documents, please answer the question.

Retrieved Documents:
{context}

Question: {question}

Answer the question using only 3-7 words. Provide only the answer, no explanation.
Answer:"""
    
    def generate_answer(self, question: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """基于检索到的文档生成答案"""
        prompt = self._create_qa_prompt(question, retrieved_docs)
        
        try:
            return self.llm_model.generate_single(prompt)
        except Exception as e:
            self.logger.error(f"[RAG系统] 问题LLM调用失败: {e}")
            return f"Error generating answer: {str(e)}"
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """完整的问答流程"""
        retrieved_docs = self.retrieve(question)
        answer = self.generate_answer(question, retrieved_docs)
        
        # 构建增强的返回结果，包含检索详情
        result = {
            'question': question,
            'answer': answer,
            'retrieved_documents': retrieved_docs,
            'num_retrieved': len(retrieved_docs),
            'retrieval_details': self._get_retrieval_summary(retrieved_docs)
        }
        
        return result
    
    def answer_questions_batch(self, questions: List[str]) -> List[Dict[str, Any]]:
        """批量问答流程"""
        if not questions:
            return []
            
        self.logger.info(f"[RAG系统] 批量处理 {len(questions)} 个问题")
        
        # 批量检索
        all_retrieved_docs = [self.retrieve(question) for question in questions]
        
        # 构建批量提示词
        prompts = [self._create_qa_prompt(question, retrieved_docs) 
                   for question, retrieved_docs in zip(questions, all_retrieved_docs)]
        
        # 批量调用LLM生成答案
        try:
            answers = self.llm_model.generate_batch(prompts)
        except Exception as e:
            self.logger.error(f"[RAG系统] 批量LLM调用失败: {e}")
            answers = [f"Error generating answer: {str(e)}"] * len(questions)
        
        # 构建结果
        results = []
        for question, answer, retrieved_docs in zip(questions, answers, all_retrieved_docs):
            result = {
                'question': question,
                'answer': answer,
                'retrieved_documents': retrieved_docs,
                'num_retrieved': len(retrieved_docs),
                'retrieval_details': self._get_retrieval_summary(retrieved_docs)
            }
            results.append(result)
        
        # 保存批量检索详情
        self._save_batch_retrieval_details(results)
        
        return results
    
    def _log_retrieval_details(self, query: str, retrieved_docs: List[Dict[str, Any]]):
        """记录单次检索的详细信息"""
        retrieval_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'num_retrieved': len(retrieved_docs),
            'retrieved_items': []
        }
        
        for i, doc in enumerate(retrieved_docs):
            # 判断是否是本体文本
            is_ontology = doc.get('doc_type') == 'ontology_summary'
            
            item_info = {
                'rank': i + 1,
                'similarity_score': doc.get('similarity_score', 0.0),
                'is_ontology_text': is_ontology,
                'doc_type': doc.get('doc_type', 'original_document'),
                'content_preview': doc.get('content', '')[:200] + '...' if len(doc.get('content', '')) > 200 else doc.get('content', ''),
                'source_info': {
                    'doc_id': doc.get('doc_id', ''),
                    'title': doc.get('title', ''),
                    'cluster_id': doc.get('cluster_id', None) if is_ontology else None
                }
            }
            retrieval_entry['retrieved_items'].append(item_info)
        
        self.retrieval_log.append(retrieval_entry)
    
    def _get_retrieval_summary(self, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取检索结果摘要"""
        if not retrieved_docs:
            return {
                'total_retrieved': 0,
                'ontology_texts': 0,
                'original_documents': 0,
                'avg_similarity': 0.0
            }
        
        ontology_count = sum(1 for doc in retrieved_docs if doc.get('doc_type') == 'ontology_summary')
        original_count = len(retrieved_docs) - ontology_count
        avg_similarity = sum(doc.get('similarity_score', 0.0) for doc in retrieved_docs) / len(retrieved_docs)
        
        return {
            'total_retrieved': len(retrieved_docs),
            'ontology_texts': ontology_count,
            'original_documents': original_count,
            'avg_similarity': avg_similarity,
            'similarity_range': {
                'min': min(doc.get('similarity_score', 0.0) for doc in retrieved_docs),
                'max': max(doc.get('similarity_score', 0.0) for doc in retrieved_docs)
            }
        }
    
    def _save_batch_retrieval_details(self, results: List[Dict[str, Any]]):
        """保存批量检索详情到JSON文件"""
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 构建详细的检索信息
        retrieval_details = {
            'timestamp': datetime.now().isoformat(),
            'total_questions': len(results),
            'retrieval_summary': {
                'total_retrievals': len(self.retrieval_log),
                'avg_docs_per_query': sum(len(entry['retrieved_items']) for entry in self.retrieval_log) / len(self.retrieval_log) if self.retrieval_log else 0
            },
            'questions_and_retrievals': []
        }
        
        # 为每个问题保存检索详情
        for result in results:
            question_info = {
                'question': result['question'],
                'answer': result['answer'],
                'retrieval_details': result['retrieval_details'],
                'retrieved_documents_analysis': []
            }
            
            # 分析检索到的文档
            for doc in result['retrieved_documents']:
                doc_analysis = {
                    'similarity_score': doc.get('similarity_score', 0.0),
                    'is_ontology_text': doc.get('doc_type') == 'ontology_summary',
                    'doc_type': doc.get('doc_type', 'original_document'),
                    'content_preview': doc.get('content', '')[:100] + '...' if len(doc.get('content', '')) > 100 else doc.get('content', ''),
                    'source_cluster_id': doc.get('cluster_id', None),
                    'doc_id': doc.get('doc_id', ''),
                    'title': doc.get('title', '')
                }
                question_info['retrieved_documents_analysis'].append(doc_analysis)
            
            retrieval_details['questions_and_retrievals'].append(question_info)
        
        # 生成文件名（包含时间戳）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"retrieval_details_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # 保存到文件
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(retrieval_details, f, ensure_ascii=False, indent=2)
            self.logger.info(f"[RAG系统] 检索详情已保存: {filepath}")
        except Exception as e:
            self.logger.error(f"[RAG系统] 保存检索详情失败: {e}")
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """获取检索统计信息"""
        if not self.retrieval_log:
            return {}
        
        total_retrievals = len(self.retrieval_log)
        total_docs_retrieved = sum(len(entry['retrieved_items']) for entry in self.retrieval_log)
        
        # 统计本体文本vs原始文档的检索情况
        ontology_retrievals = 0
        original_retrievals = 0
        
        for entry in self.retrieval_log:
            for item in entry['retrieved_items']:
                if item['is_ontology_text']:
                    ontology_retrievals += 1
                else:
                    original_retrievals += 1
        
        return {
            'total_queries': total_retrievals,
            'total_documents_retrieved': total_docs_retrieved,
            'avg_docs_per_query': total_docs_retrieved / total_retrievals if total_retrievals > 0 else 0,
            'ontology_texts_retrieved': ontology_retrievals,
            'original_documents_retrieved': original_retrievals,
            'ontology_vs_original_ratio': ontology_retrievals / original_retrievals if original_retrievals > 0 else float('inf')
        }