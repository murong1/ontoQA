#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG系统模块
负责构建检索增强生成系统
"""

import logging
import pickle
import faiss
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from .embedding_model import EmbeddingModel
from .llm_model import LLMModel


class RAGSystem:
    """RAG系统"""
    
    def __init__(self, top_k: int = 5, index_cache_dir: str = "cache/indexes"):
        """
        初始化RAG系统
        
        Args:
            top_k: 检索返回的文档数量
            index_cache_dir: 索引缓存目录
        """
        self.top_k = top_k
        self.index_cache_dir = Path(index_cache_dir)
        self.index_cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.embedding_model = EmbeddingModel()
        self.llm_model = LLMModel(llm_type="question")
        
        # 索引相关
        self.faiss_index = None
        self.documents = []
        self.summaries = {}
        self.index_documents = []
    
    def _extract_dataset_name(self, corpus_path: Path) -> str:
        """
        从语料库路径中提取数据集名称
        
        Args:
            corpus_path: 语料库文件路径
            
        Returns:
            str: 数据集名称
        """
        known_datasets = ['musique', 'quality', 'hotpot', 'nq', 'triviaqa']
        
        # 检查父目录是否是已知的数据集名称
        parent_name = corpus_path.parent.name.lower()
        if parent_name in known_datasets:
            return parent_name
        
        # 报错退出
        raise ValueError(f"无法识别数据集名称，路径: {corpus_path}，已知数据集: {known_datasets}")
    
    def _generate_cache_key(self, corpus_path: str, n_clusters: int, mode: str = "full") -> str:
        """
        生成缓存键，基于数据集路径、聚类数和实验模式
        
        Args:
            corpus_path: 语料库文件路径
            n_clusters: 聚类数量
            mode: 实验模式 ("full"包含本体总结, "documents_only"仅文档)
            
        Returns:
            str: 缓存键
        """
        # 从路径中提取数据集名称
        corpus_path_obj = Path(corpus_path)
        dataset_name = self._extract_dataset_name(corpus_path_obj)
        
        # 组合数据集名称、聚类数和模式
        if mode == "documents_only":
            cache_key = f"{dataset_name}_documents_only"
        else:
            cache_key = f"{dataset_name}_clusters_{n_clusters}"
        return cache_key
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """
        获取缓存路径
        
        Args:
            cache_key: 缓存键
            
        Returns:
            Path: 缓存目录路径
        """
        return self.index_cache_dir / cache_key
    
    def _cache_exists(self, cache_key: str) -> bool:
        """
        检查是否存在缓存的索引
        
        Args:
            cache_key: 缓存键
            
        Returns:
            bool: 是否存在缓存
        """
        cache_path = self._get_cache_path(cache_key)
        faiss_path = cache_path / "faiss_index.bin"
        data_path = cache_path / "index_data.pkl"
        return faiss_path.exists() and data_path.exists()
    
    def _load_from_cache_if_exists(self, cache_key: str) -> bool:
        """
        如果缓存存在则加载
        
        Args:
            cache_key: 缓存键
            
        Returns:
            bool: 是否从缓存加载
        """
        if self._cache_exists(cache_key):
            self.logger.info("找到缓存的索引，正在加载...")
            cache_path = self._get_cache_path(cache_key)
            self._load_cached_index(cache_path)
            self.logger.info("缓存索引加载完成")
            return True
        
        self.logger.info("未找到缓存索引，开始构建新索引...")
        return False
    
    def build_index(self, corpus: List[Dict[str, Any]], summaries: Dict[int, Dict[str, Any]], 
                   corpus_path: str, n_clusters: int):
        """
        构建包含原始文档和总结的完整索引
        
        Args:
            corpus: 原始语料库
            summaries: 聚类总结结果
            corpus_path: 语料库文件路径
            n_clusters: 聚类数量
        """
        cache_key = self._generate_cache_key(corpus_path, n_clusters, "full")
        
        if self._load_from_cache_if_exists(cache_key):
            return
        
        # 构建新索引
        self._store_data(corpus, summaries)
        index_documents = self._prepare_full_index_documents(corpus, summaries)
        self._build_and_save_index(index_documents, cache_key)
        
        self.logger.info(f"完整索引构建完成，包含 {len(index_documents)} 个文档 ({len(corpus)} 原始文档 + {len(summaries)} 聚类总结)")
    
    def build_index_documents_only(self, corpus: List[Dict[str, Any]], corpus_path: str):
        """
        构建仅包含原始文档的索引（消融实验用）
        
        Args:
            corpus: 原始语料库
            corpus_path: 语料库文件路径
        """
        cache_key = self._generate_cache_key(corpus_path, 0, "documents_only")
        
        if self._load_from_cache_if_exists(cache_key):
            return
        
        # 构建新索引
        self._store_data(corpus, {})
        index_documents = self._prepare_documents_only_index(corpus)
        self._build_and_save_index(index_documents, cache_key)
        
        self.logger.info(f"文档索引构建完成，包含 {len(index_documents)} 个原始文档（消融实验模式）")
    
    def _build_faiss_index(self, embeddings: np.ndarray):
        """
        构建FAISS索引
        
        Args:
            embeddings: 文档嵌入向量
        """
        embeddings = embeddings.astype('float32')
        dimension = embeddings.shape[1]
        
        # 使用内积搜索，L2归一化后等同于余弦相似度
        self.faiss_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)
        
        self.logger.info(f"FAISS index created with {self.faiss_index.ntotal} vectors, dimension: {dimension}")
    
    def _save_to_cache(self, cache_key: str):
        """
        保存索引到缓存
        
        Args:
            cache_key: 缓存键
        """
        if self.faiss_index is None:
            raise ValueError("No index to save. Please build index first.")
        
        cache_path = self._get_cache_path(cache_key)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # 保存FAISS索引
        faiss_path = cache_path / "faiss_index.bin"
        faiss.write_index(self.faiss_index, str(faiss_path))
        
        # 保存相关数据
        data_path = cache_path / "index_data.pkl"
        with open(data_path, 'wb') as f:
            pickle.dump({
                'index_documents': self.index_documents,
                'documents': self.documents,
                'summaries': self.summaries,
                'top_k': self.top_k
            }, f)
        
        self.logger.info(f"索引已保存到缓存: {cache_path}")
    
    def _load_cached_index(self, cache_path: Path):
        """
        从缓存加载索引
        
        Args:
            cache_path: 缓存路径
        """
        # 加载FAISS索引
        faiss_path = cache_path / "faiss_index.bin"
        self.faiss_index = faiss.read_index(str(faiss_path))
        
        # 加载相关数据
        data_path = cache_path / "index_data.pkl"
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            self.index_documents = data['index_documents']
            self.documents = data['documents']
            self.summaries = data['summaries']
            self.top_k = data['top_k']
        
        self.logger.info(f"从缓存加载索引完成，包含 {self.faiss_index.ntotal} 个向量")
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        检索相关文档
        
        Args:
            query: 查询问题
            
        Returns:
            List[Dict]: 检索到的相关文档
        """
        if self.faiss_index is None:
            raise ValueError("FAISS index has not been built yet")
        
        query_embedding = self.embedding_model.encode([query])
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), self.top_k)
        
        return [{
            **self.index_documents[idx],
            'similarity_score': float(score)
        } for score, idx in zip(scores[0], indices[0]) if idx != -1]
    
    def _store_data(self, corpus: List[Dict[str, Any]], summaries: Dict[int, Dict[str, Any]]):
        """
        存储原始数据
        
        Args:
            corpus: 原始语料库
            summaries: 聚类总结结果
        """
        self.documents = corpus.copy()
        self.summaries = summaries
    
    def _prepare_full_index_documents(self, corpus: List[Dict[str, Any]], 
                                     summaries: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        准备完整索引文档（包含原始文档和总结）
        
        Args:
            corpus: 原始语料库
            summaries: 聚类总结结果
            
        Returns:
            List[Dict]: 索引文档列表
        """
        index_documents = []
        
        # 原始文档
        for doc in corpus:
            index_documents.append({
                'type': 'original',
                'content': doc.get('context', ''),
                'title': doc.get('title', ''),
                'source': doc
            })
        
        # 总结文档
        for cluster_id, summary in summaries.items():
            index_documents.append({
                'type': 'summary',
                'content': summary.get('ontology_summary', ''),
                'title': f"Cluster {cluster_id} Summary",
                'source': summary
            })
        
        return index_documents
    
    def _prepare_documents_only_index(self, corpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        准备仅包含原始文档的索引
        
        Args:
            corpus: 原始语料库
            
        Returns:
            List[Dict]: 索引文档列表
        """
        return [{
            'type': 'original',
            'content': doc.get('context', ''),
            'title': doc.get('title', ''),
            'source': doc
        } for doc in corpus]
    
    def _build_and_save_index(self, index_documents: List[Dict[str, Any]], cache_key: str):
        """
        构建并保存索引
        
        Args:
            index_documents: 索引文档列表
            cache_key: 缓存键
        """
        texts = [doc['content'] for doc in index_documents]
        embeddings = self.embedding_model.encode(texts)
        self.index_documents = index_documents
        
        self._build_faiss_index(embeddings)
        self._save_to_cache(cache_key)
    
    def _create_qa_prompt(self, question: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        创建问答提示词
        
        Args:
            question: 问题
            retrieved_docs: 检索到的文档
            
        Returns:
            str: 提示词
        """
        context_parts = []
        for doc in retrieved_docs:
            doc_type = doc['type']
            content = doc['content']
            title = doc['title']
            score = doc['similarity_score']
            
            context_parts.append(f"[{doc_type.upper()}] {title}\n{content}\n(Relevance: {score:.3f})")
        
        context = "\n\n".join(context_parts)
        
        return f"""Based on the following retrieved documents, please answer the question.

Retrieved Documents:
{context}

Question: {question}

Answer the question using only 3-7 words. Provide only the answer, no explanation.
Answer:"""
    
    def generate_answer(self, question: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        基于检索到的文档生成答案
        
        Args:
            question: 问题
            retrieved_docs: 检索到的文档
            
        Returns:
            str: 生成的答案
        """
        prompt = self._create_qa_prompt(question, retrieved_docs)
        
        try:
            return self.llm_model.generate_single(prompt)
        except Exception as e:
            self.logger.error(f"LLM call failed for question: {e}")
            return f"Error generating answer: {str(e)}"
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        完整的问答流程
        
        Args:
            question: 问题
            
        Returns:
            Dict: 包含答案和相关信息的结果
        """
        retrieved_docs = self.retrieve(question)
        answer = self.generate_answer(question, retrieved_docs)
        
        return {
            'question': question,
            'answer': answer,
            'retrieved_documents': retrieved_docs,
            'num_retrieved': len(retrieved_docs)
        }
    
    def answer_questions_batch(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        批量问答流程
        
        Args:
            questions: 问题列表
            
        Returns:
            List[Dict]: 包含答案和相关信息的结果列表
        """
        if not questions:
            return []
            
        self.logger.info(f"Processing {len(questions)} questions in batch RAG mode")
        
        # 批量检索
        all_retrieved_docs = [self.retrieve(question) for question in questions]
        
        # 构建批量提示词
        prompts = [self._create_qa_prompt(question, retrieved_docs) 
                   for question, retrieved_docs in zip(questions, all_retrieved_docs)]
        
        # 批量调用LLM生成答案
        try:
            answers = self.llm_model.generate_batch(prompts)
        except Exception as e:
            self.logger.error(f"Batch LLM call failed: {e}")
            answers = [f"Error generating answer: {str(e)}"] * len(questions)
        
        # 构建结果
        return [{
            'question': question,
            'answer': answer,
            'retrieved_documents': retrieved_docs,
            'num_retrieved': len(retrieved_docs)
        } for question, answer, retrieved_docs in zip(questions, answers, all_retrieved_docs)]