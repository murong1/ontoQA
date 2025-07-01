#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本体处理配置模块
"""


class OntologyConfig:
    """本体处理配置类"""
    
    # 重试配置
    MAX_EXTRACTION_RETRIES = 20
    MAX_MERGE_RETRIES = 10
    
    # 并发配置
    MAX_CONCURRENT_LLM = 200
    MAX_CONCURRENT_EMBEDDING = 3
    
    # 相似度配置
    SIMILARITY_THRESHOLD = 0.85
    
    # 批处理配置
    EMBEDDING_BATCH_SIZE = 50
    SIMILARITY_BATCH_SIZE = 1000
    
    # 缓存配置
    CACHE_DIR = "cache"
    ONTOLOGY_CACHE_DIR = "ontologies"