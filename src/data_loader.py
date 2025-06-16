#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载模块
负责加载语料库和问题数据
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any


class DataLoader:
    """数据加载器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_corpus(self, corpus_path: str) -> List[Dict[str, Any]]:
        """
        加载语料库数据
        支持JSONL格式（逐行JSON对象）和单个JSON对象格式
        
        Args:
            corpus_path: 语料库文件路径
            
        Returns:
            List[Dict]: 文档列表
        """
        corpus_path = Path(corpus_path)
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
        
        self.logger.info(f"Loading corpus from {corpus_path}")
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        if not content:
            self.logger.warning("Empty corpus file")
            return []
        
        corpus = []
        
        try:
            # 尝试作为单个JSON对象解析
            data = json.loads(content)
            if isinstance(data, list):
                # 如果是列表，直接使用
                corpus = data
            elif isinstance(data, dict):
                # 如果是字典，转换为列表
                corpus = list(data.values())
            else:
                raise ValueError(f"Unsupported JSON format, expected list or dict, got {type(data)}")
                
        except json.JSONDecodeError:
            # 如果单个JSON解析失败，尝试JSONL格式（逐行JSON）
            self.logger.info("Single JSON parsing failed, trying JSONL format...")
            
            for line_num, line in enumerate(content.split('\n'), 1):
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        corpus.append(item)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON parsing error at line {line_num}: {e}")
                        self.logger.error(f"Problematic line: {line[:100]}...")
                        raise ValueError(f"Invalid JSON format at line {line_num}: {str(e)}")
        
        if not corpus:
            raise ValueError("No valid documents found in corpus file")
        
        # 验证数据格式
        for i, doc in enumerate(corpus[:5]):  # 检查前5个文档
            if not isinstance(doc, dict):
                raise ValueError(f"Document {i} is not a dictionary: {type(doc)}")
        
        self.logger.info(f"Loaded {len(corpus)} documents")
        return corpus
    
    def load_questions(self, questions_path: str) -> List[Dict[str, Any]]:
        """
        加载问题数据
        支持JSONL格式（逐行JSON对象）和单个JSON对象格式
        
        Args:
            questions_path: 问题文件路径
            
        Returns:
            List[Dict]: 问题列表
        """
        questions_path = Path(questions_path)
        if not questions_path.exists():
            raise FileNotFoundError(f"Questions file not found: {questions_path}")
        
        self.logger.info(f"Loading questions from {questions_path}")
        
        with open(questions_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        if not content:
            self.logger.warning("Empty questions file")
            return []
        
        questions = []
        
        try:
            # 尝试作为单个JSON对象解析
            data = json.loads(content)
            if isinstance(data, list):
                # 如果是列表，直接使用
                questions = data
            elif isinstance(data, dict):
                # 如果是字典，转换为列表
                questions = list(data.values())
            else:
                raise ValueError(f"Unsupported JSON format, expected list or dict, got {type(data)}")
                
        except json.JSONDecodeError:
            # 如果单个JSON解析失败，尝试JSONL格式（逐行JSON）
            self.logger.info("Single JSON parsing failed, trying JSONL format...")
            
            for line_num, line in enumerate(content.split('\n'), 1):
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        questions.append(item)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON parsing error at line {line_num}: {e}")
                        self.logger.error(f"Problematic line: {line[:100]}...")
                        raise ValueError(f"Invalid JSON format at line {line_num}: {str(e)}")
        
        if not questions:
            raise ValueError("No valid questions found in questions file")
        
        # 验证数据格式
        for i, question in enumerate(questions[:5]):  # 检查前5个问题
            if not isinstance(question, dict):
                raise ValueError(f"Question {i} is not a dictionary: {type(question)}")
        
        self.logger.info(f"Loaded {len(questions)} questions")
        return questions