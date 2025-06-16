#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM模型调用模块
支持OpenAI API调用
"""

import json
import logging
import requests
from typing import List, Union, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random


class LLMModel:
    """LLM模型调用器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, llm_type: str = "question"):
        """
        初始化LLM模型
        
        Args:
            config: 配置字典，如果为None则使用默认配置
            llm_type: LLM类型，"question"或"summary"
        """
        self.logger = logging.getLogger(__name__)
        
        if config is None:
            from config import Config
            if llm_type == "question":
                self.config = Config.get_question_llm_config()
            else:
                self.config = Config.get_summary_llm_config()
        else:
            self.config = config
            
        self.provider = self.config.get('provider', 'openai')
        self.api_url = self.config.get('api_url')
        self.api_key = self.config.get('api_key')
        self.model = self.config.get('model')
        self.max_tokens = self.config.get('max_tokens', 1000)
        self.max_workers = self.config.get('max_workers', 10)
        self.temperature = self.config.get('temperature', 0.7)
        
        # 重试配置
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 1.0)  # 基础延迟秒数
        self.retry_backoff = self.config.get('retry_backoff', 2.0)  # 退避倍数
        self.retry_jitter = self.config.get('retry_jitter', True)  # 是否添加随机抖动
        
        self.logger.info(f"Initialized LLM model: {self.model} ({llm_type})")
    
    def _should_retry(self, error: Exception) -> bool:
        """
        判断是否应该重试
        
        Args:
            error: 异常对象
            
        Returns:
            bool: 是否应该重试
        """
        if isinstance(error, requests.exceptions.Timeout):
            return True
        elif isinstance(error, requests.exceptions.ConnectionError):
            return True
        elif isinstance(error, requests.exceptions.HTTPError):
            # 对于HTTP错误，只重试5xx服务器错误和429 Too Many Requests
            if hasattr(error, 'response') and error.response is not None:
                status_code = error.response.status_code
                return status_code >= 500 or status_code == 429
            return False
        elif isinstance(error, (json.JSONDecodeError, KeyError)):
            # 响应格式错误通常不应该重试，可能是API版本不兼容
            return False
        return False
    
    def _get_retry_delay(self, attempt: int) -> float:
        """
        计算重试延迟时间
        
        Args:
            attempt: 当前重试次数（从0开始）
            
        Returns:
            float: 延迟秒数
        """
        # 指数退避: base_delay * (backoff ^ attempt)
        delay = self.retry_delay * (self.retry_backoff ** attempt)
        
        # 添加随机抖动避免惊群效应
        if self.retry_jitter:
            jitter = random.uniform(0.1, 0.5) * delay
            delay += jitter
        
        return delay
    
    def _call_api(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        调用LLM API（带重试机制）
        
        Args:
            messages: 消息列表
            **kwargs: 额外参数
            
        Returns:
            str: 生成的回复
        """
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        data = {
            'model': self.model,
            'messages': messages,
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            'temperature': kwargs.get('temperature', self.temperature)
        }
        
        last_error = None
        
        for attempt in range(self.max_retries + 1):  # +1 因为第一次不算重试
            try:
                response = requests.post(
                    f"{self.api_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                response.raise_for_status()
                
                result = response.json()
                
                # 验证响应格式
                if 'choices' not in result or not result['choices']:
                    raise ValueError("API response missing 'choices' field or empty choices")
                
                choice = result['choices'][0]
                if 'message' not in choice or 'content' not in choice['message']:
                    raise ValueError("API response missing message content")
                
                content = choice['message']['content']
                if content is None:
                    raise ValueError("API response content is None")
                
                # 成功返回
                if attempt > 0:
                    self.logger.info(f"API call succeeded after {attempt} retries")
                
                return content
                
            except requests.exceptions.Timeout as e:
                last_error = e
                error_msg = f"API request timeout after 30s"
                self.logger.warning(f"Attempt {attempt + 1}: {error_msg}")
                
            except requests.exceptions.ConnectionError as e:
                last_error = e
                error_msg = f"API connection error: {str(e)}"
                self.logger.warning(f"Attempt {attempt + 1}: {error_msg}")
                
            except requests.exceptions.HTTPError as e:
                last_error = e
                status_code = e.response.status_code if e.response else 'unknown'
                response_text = e.response.text if e.response else 'no response'
                error_msg = f"API HTTP error {status_code}: {response_text}"
                self.logger.warning(f"Attempt {attempt + 1}: {error_msg}")
                
            except requests.exceptions.RequestException as e:
                last_error = e
                error_msg = f"API request error: {str(e)}"
                self.logger.warning(f"Attempt {attempt + 1}: {error_msg}")
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                last_error = e
                error_msg = f"API response format error: {str(e)}"
                self.logger.error(f"Attempt {attempt + 1}: {error_msg}")
                # 响应格式错误通常不需要重试
                if not self._should_retry(e):
                    break
            
            # 检查是否应该重试
            if not self._should_retry(last_error):
                self.logger.error(f"Error not retryable, aborting: {last_error}")
                break
                
            # 如果不是最后一次尝试，等待后重试
            if attempt < self.max_retries:
                delay = self._get_retry_delay(attempt)
                self.logger.info(f"Retrying in {delay:.2f} seconds... (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(delay)
        
        # 所有重试都失败了
        error_msg = f"API call failed after {self.max_retries + 1} attempts. Last error: {last_error}"
        self.logger.error(error_msg)
        raise RuntimeError(error_msg) from last_error
    
    def generate_single(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """
        生成单个回复
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            **kwargs: 额外参数
            
        Returns:
            str: 生成的回复
        """
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user", 
            "content": prompt
        })
        
        return self._call_api(messages, **kwargs)
    
    def generate_batch(self, prompts: List[str], system_prompt: str = None, **kwargs) -> List[str]:
        """
        批量生成回复
        
        Args:
            prompts: 提示列表
            system_prompt: 系统提示
            **kwargs: 额外参数
            
        Returns:
            List[str]: 生成的回复列表
        """
        if not prompts:
            return []
        
        results = [None] * len(prompts)
        
        self.logger.info(f"Processing {len(prompts)} prompts with {self.max_workers} workers")
        
        def process_prompt(idx_prompt):
            idx, prompt = idx_prompt
            try:
                messages = []
                
                if system_prompt:
                    messages.append({
                        "role": "system",
                        "content": system_prompt
                    })
                
                messages.append({
                    "role": "user",
                    "content": prompt
                })
                
                result = self._call_api(messages, **kwargs)
                return idx, result
            except Exception as e:
                self.logger.error(f"Error processing prompt {idx}: {e}")
                return idx, f"Error: {str(e)}"
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(prompts))) as executor:
            futures = [executor.submit(process_prompt, (i, prompt)) for i, prompt in enumerate(prompts)]
            
            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    self.logger.error(f"Future processing failed: {e}")
        
        return results
    
    def generate(self, prompts: Union[str, List[str]], system_prompt: str = None, **kwargs) -> Union[str, List[str]]:
        """
        生成回复（支持单个或批量）
        
        Args:
            prompts: 单个提示或提示列表
            system_prompt: 系统提示
            **kwargs: 额外参数
            
        Returns:
            Union[str, List[str]]: 单个回复或回复列表
        """
        if isinstance(prompts, str):
            return self.generate_single(prompts, system_prompt, **kwargs)
        else:
            return self.generate_batch(prompts, system_prompt, **kwargs)