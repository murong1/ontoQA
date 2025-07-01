#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding模型调用模块
支持API和本地模型调用，优化了批量和高并发处理
"""

import json
import logging
import requests
import time
import asyncio
import aiohttp
from typing import List, Union, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from threading import Lock, BoundedSemaphore
from queue import Queue, Empty
import threading
from sentence_transformers import SentenceTransformer
from modelscope import snapshot_download


class EmbeddingModel:
    """高性能Embedding模型调用器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化embedding模型
        
        Args:
            config: 配置字典，如果为None则使用默认配置
        """
        self.logger = logging.getLogger(__name__)
        
        if config is None:
            from config import Config
            self.config = Config.get_embedding_config()
        else:
            self.config = config
            
        self.provider = self.config.get('provider', 'api')
        
        # API配置
        if self.provider == 'api':
            self.api_url = self.config.get('api_url')
            self.api_key = self.config.get('api_key', 'EMPTY')
            self.model_name = self.config.get('model_name')
            self.timeout = self.config.get('timeout', 30)
            
        # 并发控制配置
        self.batch_size = self.config.get('batch_size', 32)
        self.max_length = self.config.get('max_length', 512)
        self.max_workers = self.config.get('max_workers', 20)
        self.rate_limit = self.config.get('rate_limit', 10)  # 每秒最大请求数
        self.adaptive_batch = self.config.get('adaptive_batch', True)  # 动态批次大小
        
        # 重试配置
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 1.0)
        self.backoff_factor = self.config.get('backoff_factor', 2.0)
        
        # 并发控制
        self._rate_limiter = BoundedSemaphore(self.rate_limit)
        self._result_lock = Lock()
        self._request_times = Queue()
        
        # 性能监控
        self._success_count = 0
        self._failure_count = 0
        self._total_latency = 0.0
        self._adaptive_sizes = []
        
        # 初始化本地模型
        self.local_model = None
        if self.provider == 'local':
            model_name = self.config.get('model_name', 'BAAI/bge-m3')
            self.logger.info(f"[嵌入模型] 从魔搭社区加载本地模型: {model_name}")
            
            try:
                # 从魔搭社区下载模型
                model_dir = snapshot_download('BAAI/bge-m3', cache_dir='./models')
                self.logger.info(f"[嵌入模型] 模型下载到: {model_dir}")
                
                # 使用本地路径加载模型
                self.local_model = SentenceTransformer(model_dir)
                self.logger.info(f"[嵌入模型] 从魔搭社区加载模型成功")
                
            except Exception as e:
                self.logger.warning(f"[嵌入模型] 从魔搭社区加载失败: {e}")
                self.logger.info(f"[嵌入模型] 降级使用HuggingFace: {model_name}")
                self.local_model = SentenceTransformer(model_name)
                self.logger.info(f"[嵌入模型] 从HuggingFace加载模型成功")
        
        self.logger.info(f"[嵌入模型] 初始化完成: provider={self.provider}, "
                        f"batch_size={self.batch_size}, max_workers={self.max_workers}, "
                        f"rate_limit={self.rate_limit}, adaptive_batch={self.adaptive_batch}")
    
    def _truncate_text(self, text: str) -> str:
        """截断文本到最大长度"""
        return text[:self.max_length] if len(text) > self.max_length else text
    
    def _rate_limit_wait(self):
        """速率限制控制"""
        current_time = time.time()
        
        # 清理过期的请求时间记录
        while not self._request_times.empty():
            try:
                old_time = self._request_times.get_nowait()
                if current_time - old_time > 1.0:  # 超过1秒的记录清理
                    continue
                else:
                    self._request_times.put(old_time)  # 放回去
                    break
            except Empty:
                break
        
        # 如果队列满了，等待
        if self._request_times.qsize() >= self.rate_limit:
            wait_time = 1.0 / self.rate_limit
            time.sleep(wait_time)
        
        self._request_times.put(current_time)
    
    def _calculate_adaptive_batch_size(self, remaining_texts: int) -> int:
        """动态计算批次大小"""
        if not self.adaptive_batch:
            return min(self.batch_size, remaining_texts)
        
        # 基于成功率和延迟调整批次大小
        if self._success_count + self._failure_count < 10:
            return min(self.batch_size, remaining_texts)
        
        success_rate = self._success_count / (self._success_count + self._failure_count)
        avg_latency = self._total_latency / max(self._success_count, 1)
        
        # 根据成功率调整
        if success_rate < 0.8:
            # 成功率低，减小批次
            adjusted_size = max(1, self.batch_size // 2)
        elif success_rate > 0.95 and avg_latency < 2.0:
            # 成功率高且延迟低，增大批次
            adjusted_size = min(self.batch_size * 2, 64)
        else:
            adjusted_size = self.batch_size
        
        return min(adjusted_size, remaining_texts)
    
    def _smart_batch_split(self, texts: List[str]) -> List[List[str]]:
        """智能批次分割"""
        batches = []
        i = 0
        
        while i < len(texts):
            remaining = len(texts) - i
            batch_size = self._calculate_adaptive_batch_size(remaining)
            
            batch = texts[i:i + batch_size]
            batches.append(batch)
            
            self._adaptive_sizes.append(batch_size)
            i += batch_size
        
        self.logger.info(f"[嵌入模型] 将 {len(texts)} 个文本分为 {len(batches)} 个自适应批次")
        return batches
    
    def _call_api_batch_with_retry(self, texts: List[str]) -> List[List[float]]:
        """
        带重试机制的API调用
        
        Args:
            texts: 文本列表
            
        Returns:
            List[List[float]]: embedding向量列表
        """
        if not texts:
            return []
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # 速率限制
                self._rate_limit_wait()
                
                start_time = time.time()
                result = self._call_api_batch(texts)
                latency = time.time() - start_time
                
                # 更新性能统计
                with self._result_lock:
                    self._success_count += 1
                    self._total_latency += latency
                
                self.logger.debug(f"[嵌入模型] 批量API调用成功，耗时 {latency:.2f}s")
                return result
                
            except Exception as e:
                last_exception = e
                with self._result_lock:
                    self._failure_count += 1
                
                if attempt < self.max_retries:
                    delay = self.retry_delay * (self.backoff_factor ** attempt)
                    self.logger.warning(f"[嵌入模型] 批量API调用失败 (第{attempt + 1}/{self.max_retries + 1}次): {e}")
                    self.logger.info(f"[嵌入模型] {delay:.2f}秒后重试...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"[嵌入模型] 批量API调用在{self.max_retries + 1}次尝试后失败")
        
        raise last_exception
    
    def _call_api_batch(self, texts: List[str]) -> List[List[float]]:
        """
        调用API进行batch embedding
        
        Args:
            texts: 文本列表
            
        Returns:
            List[List[float]]: embedding向量列表
        """
        payload = {
            'input': texts,
            'model': self.model_name
        }
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        
        # 详细的HTTP错误处理
        if response.status_code == 429:
            raise RuntimeError("API rate limit exceeded")
        elif response.status_code == 401:
            raise RuntimeError("API authentication failed")
        elif response.status_code == 413:
            raise RuntimeError("Request payload too large")
        elif response.status_code >= 500:
            raise RuntimeError(f"API server error: {response.status_code}")
        
        response.raise_for_status()
        
        try:
            result = response.json()
            return self._extract_embeddings(result, len(texts))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response: {e}")
        except Exception as e:
            raise RuntimeError(f"Response parsing error: {e}")
    
    def _extract_embeddings(self, response: Dict[str, Any], expected_count: int) -> List[List[float]]:
        """
        从 API 响应中提取 embedding
        
        Args:
            response: API 响应
            expected_count: 期望的 embedding 数量
            
        Returns:
            List[List[float]]: embedding 向量列表
        """
        try:
            if 'data' not in response:
                raise ValueError(f"API response missing 'data' field, available keys: {list(response.keys())}")
            
            data = response['data']
            if not isinstance(data, list):
                raise ValueError(f"'data' field should be a list, got {type(data)}")
            
            embeddings = []
            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    raise ValueError(f"Data item {i} should be a dict, got {type(item)}")
                
                if 'embedding' not in item:
                    available_keys = list(item.keys())
                    raise ValueError(f"API response item {i} missing 'embedding' field, available keys: {available_keys}")
                
                embedding = item['embedding']
                if not isinstance(embedding, list):
                    raise ValueError(f"Embedding {i} should be a list, got {type(embedding)}")
                
                # 验证embedding向量
                if not embedding:
                    raise ValueError(f"Embedding {i} is empty")
                
                # 检查embedding是否包含有效数值
                try:
                    float_embedding = [float(x) for x in embedding]
                    embeddings.append(float_embedding)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Embedding {i} contains invalid values: {e}")
            
            if len(embeddings) != expected_count:
                raise ValueError(f"Expected {expected_count} embeddings, got {len(embeddings)}")
            
            # 验证所有embedding维度一致
            if embeddings:
                first_dim = len(embeddings[0])
                for i, emb in enumerate(embeddings[1:], 1):
                    if len(emb) != first_dim:
                        raise ValueError(f"Embedding {i} dimension {len(emb)} != first embedding dimension {first_dim}")
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"[嵌入模型] 提取embedding失败: {e}")
            self.logger.debug(f"[嵌入模型] 响应结构: {json.dumps(response, indent=2)[:500]}...")
            raise
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        高性能批量编码文本（智能批处理+错误恢复）
        
        Args:
            texts: 文本列表
            
        Returns:
            np.ndarray: embedding矩阵
        """
        if not texts:
            return np.array([])
        
        # 重置性能统计
        self._success_count = 0
        self._failure_count = 0
        self._total_latency = 0.0
        self._adaptive_sizes = []
        
        truncated_texts = [self._truncate_text(text) for text in texts]
        
        self.logger.info(f"[嵌入模型] 使用自适应批次处理 {len(truncated_texts)} 个文本")
        
        if self.provider == 'api':
            return self._process_batches_with_recovery(truncated_texts)
        elif self.provider == 'local':
            return self._process_local_batches(truncated_texts)
        else:
            raise NotImplementedError(f"Provider {self.provider} not implemented")
    
    def _process_batches_with_recovery(self, texts: List[str]) -> np.ndarray:
        """
        智能批处理+错误恢复
        
        Args:
            texts: 截断后的文本列表
            
        Returns:
            np.ndarray: embedding矩阵
        """
        batches = self._smart_batch_split(texts)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有批次任务
            futures_to_info = {}
            for batch_idx, batch in enumerate(batches):
                start_pos = sum(len(batches[i]) for i in range(batch_idx))
                future = executor.submit(self._call_api_batch_with_retry, batch)
                futures_to_info[future] = {
                    'batch_idx': batch_idx,
                    'start_pos': start_pos,
                    'batch_size': len(batch),
                    'batch': batch
                }
            
            # 收集结果并处理错误
            batch_results = {}
            failed_batches = []
            
            for future in as_completed(futures_to_info.keys()):
                info = futures_to_info[future]
                batch_idx = info['batch_idx']
                
                try:
                    batch_embeddings = future.result()
                    batch_results[info['start_pos']] = batch_embeddings
                    self.logger.debug(f"[嵌入模型] 完成批次 {batch_idx + 1}/{len(batches)} "
                                    f"（大小: {info['batch_size']}）")
                except Exception as e:
                    self.logger.error(f"[嵌入模型] 批次 {batch_idx + 1} 永久性失败: {e}")
                    failed_batches.append({
                        'batch_idx': batch_idx,
                        'error': str(e),
                        'texts': info['batch']
                    })
            
            # 尝试恢复失败的批次
            if failed_batches:
                self.logger.warning(f"[嵌入模型] 尝试恢复 {len(failed_batches)} 个失败批次")
                recovered_results = self._recover_failed_batches(failed_batches)
                
                for start_pos, embeddings in recovered_results.items():
                    batch_results[start_pos] = embeddings
            
            # 验证所有批次都成功
            total_expected = len(texts)
            total_processed = sum(len(embs) for embs in batch_results.values())
            
            if total_processed != total_expected:
                missing_count = total_expected - total_processed
                raise RuntimeError(f"Failed to process {missing_count} texts out of {total_expected}")
            
            # 按顺序合并结果
            all_embeddings = []
            current_pos = 0
            
            while current_pos < len(texts):
                if current_pos not in batch_results:
                    raise RuntimeError(f"Missing batch result for position {current_pos}")
                
                batch_embeddings = batch_results[current_pos]
                all_embeddings.extend(batch_embeddings)
                current_pos += len(batch_embeddings)
            
            embeddings_array = np.array(all_embeddings)
            
            # 性能统计
            success_rate = self._success_count / max(self._success_count + self._failure_count, 1)
            avg_latency = self._total_latency / max(self._success_count, 1)
            
            self.logger.info(f"[嵌入模型] 处理完成: shape={embeddings_array.shape}, "
                           f"success_rate={success_rate:.2%}, avg_latency={avg_latency:.2f}s, "
                           f"adaptive_sizes={self._adaptive_sizes}")
            
            return embeddings_array
    
    def _recover_failed_batches(self, failed_batches: List[Dict]) -> Dict[int, List[List[float]]]:
        """
        恢复失败的批次（单个文本重试）
        
        Args:
            failed_batches: 失败的批次信息列表
            
        Returns:
            Dict[int, List[List[float]]]: 恢复的结果
        """
        recovered_results = {}
        
        for batch_info in failed_batches:
            batch_texts = batch_info['texts']
            self.logger.info(f"[嵌入模型] 尝试批次 {batch_info['batch_idx'] + 1} 的单个文本恢复")
            
            individual_embeddings = []
            
            # 逐个重试文本
            for i, text in enumerate(batch_texts):
                try:
                    # 单个文本重试
                    single_embedding = self._call_api_batch_with_retry([text])
                    individual_embeddings.extend(single_embedding)
                    self.logger.debug(f"[嵌入模型] 恢复文本 {i + 1}/{len(batch_texts)}")
                except Exception as e:
                    self.logger.error(f"[嵌入模型] 恢复文本 {i + 1} 失败: {e}")
                    # 对于完全失败的文本，可以考虑使用零向量或跳过
                    raise RuntimeError(f"Unable to recover text: {text[:100]}...")
            
            # 计算起始位置
            start_pos = sum(len(failed_batches[j]['texts']) 
                          for j in range(batch_info['batch_idx']))
            
            recovered_results[start_pos] = individual_embeddings
            self.logger.info(f"[嵌入模型] 成功恢复批次 {batch_info['batch_idx'] + 1}")
        
        return recovered_results
    
    def _process_local_batches(self, texts: List[str]) -> np.ndarray:
        """
        本地模型批处理
        
        Args:
            texts: 文本列表
            
        Returns:
            np.ndarray: embedding矩阵
        """
        if self.local_model is None:
            raise RuntimeError("Local model not initialized")
        
        self.logger.info(f"[嵌入模型] 使用本地模型处理 {len(texts)} 个文本")
        
        try:
            start_time = time.time()
            
            # 使用sentence-transformers直接批处理
            embeddings = self.local_model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # 归一化embedding向量
            )
            
            latency = time.time() - start_time
            
            # 更新性能统计
            with self._result_lock:
                self._success_count += 1
                self._total_latency += latency
            
            self.logger.info(f"[嵌入模型] 本地处理完成: shape={embeddings.shape}, "
                           f"time={latency:.2f}s, avg_time_per_text={latency/len(texts):.4f}s")
            
            return embeddings
            
        except Exception as e:
            with self._result_lock:
                self._failure_count += 1
            self.logger.error(f"[嵌入模型] 本地模型处理失败: {e}")
            raise
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        编码单个文本
        
        Args:
            text: 输入文本
            
        Returns:
            np.ndarray: embedding向量
        """
        embeddings = self.encode_batch([text])
        return embeddings[0] if len(embeddings) > 0 else np.array([])
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        编码文本（支持单个或批量）
        
        Args:
            texts: 单个文本或文本列表
            
        Returns:
            np.ndarray: embedding向量或矩阵
        """
        if isinstance(texts, str):
            return self.encode_single(texts)
        else:
            return self.encode_batch(texts)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计信息
        
        Returns:
            Dict[str, Any]: 性能统计数据
        """
        total_requests = self._success_count + self._failure_count
        success_rate = self._success_count / max(total_requests, 1)
        avg_latency = self._total_latency / max(self._success_count, 1)
        
        return {
            'total_requests': total_requests,
            'success_count': self._success_count,
            'failure_count': self._failure_count,
            'success_rate': success_rate,
            'average_latency': avg_latency,
            'adaptive_batch_sizes': self._adaptive_sizes.copy() if self._adaptive_sizes else []
        }
    
    def reset_stats(self):
        """重置性能统计"""
        self._success_count = 0
        self._failure_count = 0
        self._total_latency = 0.0
        self._adaptive_sizes = []
        
        # 清空请求时间队列
        while not self._request_times.empty():
            try:
                self._request_times.get_nowait()
            except Empty:
                break