#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件
"""

import os
from typing import Dict, Any


class Config:
    """项目配置类"""
    MIOASHU = "测试部分"
    
    # 数据配置
    DEFAULT_CORPUS_PATH = "datasets/musique/test_corpus_100.json"
    DEFAULT_QUESTIONS_PATH = "datasets/musique/test_questions_20.json"
    
    # 聚类配置
    DEFAULT_N_CLUSTERS = 5  # 默认聚类数，实际运行时会根据文档数量动态调整为文档数的1/5到1/10
    
    # RAG配置
    DEFAULT_TOP_K = 15
    
    # 问答LLM配置
    QUESTION_LLM_PROVIDER = "openai"  # 或 "local"
    QUESTION_LLM_API_URL = "https://chatapi.zjt66.top/v1"
    QUESTION_LLM_API_KEY = "sk-d2JVZ12Td33eVv54a5ykpuj2UttHMepiN3P69JRRxwRbSVWL"
    QUESTION_LLM_MODEL = "gpt-4o-mini"
    QUESTION_LLM_MAX_TOKENS = 50
    QUESTION_LLM_MAX_WORKERS = 15
    QUESTION_LLM_TEMPERATURE = 1
    QUESTION_LLM_MAX_RETRIES = 200
    QUESTION_LLM_RETRY_DELAY = 1.0
    QUESTION_LLM_RETRY_BACKOFF = 2.0
    QUESTION_LLM_RETRY_JITTER = True
    
    # 本体总结LLM配置
    SUMMARY_LLM_PROVIDER = "openai"
    SUMMARY_LLM_API_URL = "https://chatapi.zjt66.top/v1"
    SUMMARY_LLM_API_KEY = "sk-d2JVZ12Td33eVv54a5ykpuj2UttHMepiN3P69JRRxwRbSVWL"
    SUMMARY_LLM_MODEL = "gpt-4o-mini"
    SUMMARY_LLM_MAX_TOKENS = 2000
    SUMMARY_LLM_MAX_WORKERS = 15
    SUMMARY_LLM_TEMPERATURE = 1
    SUMMARY_LLM_MAX_RETRIES = 200
    SUMMARY_LLM_RETRY_DELAY = 1.0
    SUMMARY_LLM_RETRY_BACKOFF = 2.0
    SUMMARY_LLM_RETRY_JITTER = True
    
    # Embedding配置
    EMBEDDING_PROVIDER = "local"  # 或 "api"
    EMBEDDING_API_URL = "http://localhost:8000/v1/embeddings"
    EMBEDDING_API_KEY = "EMPTY"
    EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
    EMBEDDING_BATCH_SIZE = 32
    EMBEDDING_MAX_LENGTH = 512
    EMBEDDING_TIMEOUT = 180  # API请求超时时间（秒）
    EMBEDDING_MAX_WORKERS = 1  # 并行处理的工作线程数
    
    # 输出配置
    DEFAULT_OUTPUT_DIR = "results"
    LOG_FILE = "experiment.log"
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """获取配置字典"""
        return {
            'corpus_path': cls.DEFAULT_CORPUS_PATH,
            'questions_path': cls.DEFAULT_QUESTIONS_PATH,
            'n_clusters': cls.DEFAULT_N_CLUSTERS,
            'top_k': cls.DEFAULT_TOP_K,
            'output_dir': cls.DEFAULT_OUTPUT_DIR
        }
    
    @classmethod
    def get_embedding_config(cls) -> Dict[str, Any]:
        """获取embedding配置字典"""
        return {
            'provider': cls.EMBEDDING_PROVIDER,
            'api_url': cls.EMBEDDING_API_URL,
            'api_key': cls.EMBEDDING_API_KEY,
            'model_name': cls.EMBEDDING_MODEL_NAME,
            'batch_size': cls.EMBEDDING_BATCH_SIZE,
            'max_length': cls.EMBEDDING_MAX_LENGTH,
            'timeout': cls.EMBEDDING_TIMEOUT,
            'max_workers': cls.EMBEDDING_MAX_WORKERS
        }
    
    @classmethod
    def get_question_llm_config(cls) -> Dict[str, Any]:
        """获取问答LLM配置字典"""
        return {
            'provider': cls.QUESTION_LLM_PROVIDER,
            'api_url': cls.QUESTION_LLM_API_URL,
            'api_key': cls.QUESTION_LLM_API_KEY,
            'model': cls.QUESTION_LLM_MODEL,
            'max_tokens': cls.QUESTION_LLM_MAX_TOKENS,
            'max_workers': cls.QUESTION_LLM_MAX_WORKERS,
            'temperature': cls.QUESTION_LLM_TEMPERATURE,
            'max_retries': cls.QUESTION_LLM_MAX_RETRIES,
            'retry_delay': cls.QUESTION_LLM_RETRY_DELAY,
            'retry_backoff': cls.QUESTION_LLM_RETRY_BACKOFF,
            'retry_jitter': cls.QUESTION_LLM_RETRY_JITTER
        }
    
    @classmethod
    def get_summary_llm_config(cls) -> Dict[str, Any]:
        """获取总结LLM配置字典"""
        return {
            'provider': cls.SUMMARY_LLM_PROVIDER,
            'api_url': cls.SUMMARY_LLM_API_URL,
            'api_key': cls.SUMMARY_LLM_API_KEY,
            'model': cls.SUMMARY_LLM_MODEL,
            'max_tokens': cls.SUMMARY_LLM_MAX_TOKENS,
            'max_workers': cls.SUMMARY_LLM_MAX_WORKERS,
            'temperature': cls.SUMMARY_LLM_TEMPERATURE,
            'max_retries': cls.SUMMARY_LLM_MAX_RETRIES,
            'retry_delay': cls.SUMMARY_LLM_RETRY_DELAY,
            'retry_backoff': cls.SUMMARY_LLM_RETRY_BACKOFF,
            'retry_jitter': cls.SUMMARY_LLM_RETRY_JITTER
        }
    
