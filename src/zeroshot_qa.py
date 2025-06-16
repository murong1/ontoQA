#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-shot问答模块
直接使用LLM进行问答，无需检索，用于建立baseline
"""

import logging
from typing import List, Dict, Any
from .llm_model import LLMModel


class ZeroshotQA:
    """Zero-shot问答系统"""
    
    def __init__(self):
        """
        初始化Zero-shot问答系统
        """
        self.logger = logging.getLogger(__name__)
        self.llm_model = LLMModel(llm_type="question")
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        直接使用LLM回答问题（无检索）
        
        Args:
            question: 问题
            
        Returns:
            Dict: 包含答案和相关信息的结果
        """
        # 构建Zero-shot提示词
        prompt = f"""Please answer the following question based on your knowledge.

Question: {question}

Answer the question using only 3-7 words. Provide only the answer, no explanation."""
        
        # 调用LLM生成答案
        try:
            answer = self.llm_model.generate_single(prompt)
        except Exception as e:
            self.logger.error(f"LLM call failed for zero-shot question: {e}")
            answer = f"Error generating answer: {str(e)}"
        
        return {
            'question': question,
            'answer': answer,
            'mode': 'zeroshot',
            'retrieved_documents': [],
            'num_retrieved': 0
        }
    
    def answer_questions_batch(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        批量回答问题（使用并发）
        
        Args:
            questions: 问题列表
            
        Returns:
            List[Dict]: 答案结果列表
        """
        if not questions:
            return []
            
        self.logger.info(f"Processing {len(questions)} questions in batch mode")
        
        # 构建批量提示词
        prompts = []
        for question in questions:
            prompt = f"""Please answer the following question based on your knowledge.

Question: {question}

Answer the question using only 3-7 words. Provide only the answer, no explanation."""
            prompts.append(prompt)
        
        # 批量调用LLM
        answers = self.llm_model.generate_batch(prompts)
        
        # 构建结果
        results = []
        for question, answer in zip(questions, answers):
            result = {
                'question': question,
                'answer': answer,
                'mode': 'zeroshot',
                'retrieved_documents': [],
                'num_retrieved': 0
            }
            results.append(result)
        
        return results