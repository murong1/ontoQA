#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重试工具模块
提供重试装饰器和相关工具函数
"""

import functools
import logging
import json
from typing import Callable, Any, Type, Tuple


def retry_with_logging(max_retries: int, 
                      exception_types: Tuple[Type[Exception], ...] = (Exception,),
                      logger_name: str = __name__) -> Callable:
    """
    重试装饰器，带日志记录
    
    Args:
        max_retries: 最大重试次数
        exception_types: 需要重试的异常类型
        logger_name: 日志记录器名称
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger = logging.getLogger(logger_name)
            
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"[重试成功] {func.__name__} 第 {attempt + 1} 次尝试成功")
                    return result
                    
                except exception_types as e:
                    if attempt == max_retries - 1:
                        logger.error(f"[重试失败] {func.__name__} 达到最大重试次数 {max_retries}")
                        raise RuntimeError(f"{func.__name__} 经过 {max_retries} 次重试仍然失败: {e}") from e
                    else:
                        logger.warning(f"[重试] {func.__name__} 第 {attempt + 1} 次尝试失败: {e}")
                        continue
                        
            raise RuntimeError(f"{func.__name__} 重试逻辑异常")
            
        return wrapper
    return decorator


def is_valid_json_response(response: str) -> bool:
    """
    检查响应是否包含有效的JSON
    
    Args:
        response: 待检查的响应字符串
        
    Returns:
        bool: 是否包含有效JSON
    """
    try:
        # 尝试提取JSON部分
        import re
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response.strip()
            
        parsed = json.loads(json_str)
        return isinstance(parsed, dict)
    except (json.JSONDecodeError, AttributeError):
        return False


def extract_valid_content(response: str, required_keys: list = None) -> bool:
    """
    检查响应是否包含所需的内容
    
    Args:
        response: 响应字符串
        required_keys: 必需的键列表
        
    Returns:
        bool: 是否包含有效内容
    """
    if not response or not response.strip():
        return False
        
    if required_keys:
        try:
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                parsed = json.loads(json_str)
                return all(key in parsed for key in required_keys)
        except (json.JSONDecodeError, AttributeError):
            pass
            
    return True  # 如果没有特定要求，只要有内容就认为有效