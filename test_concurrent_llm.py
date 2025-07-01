#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
200并发worker LLM请求测试
用于验证高并发场景下LLM调用的稳定性
"""

import sys
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import threading
from datetime import datetime
import json

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.llm_model import LLMModel
from config import Config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('concurrent_test.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class ConcurrentLLMTester:
    """并发LLM测试器"""
    
    def __init__(self, max_workers: int = 200):
        """
        初始化测试器
        
        Args:
            max_workers: 最大并发worker数量
        """
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        
        # 获取配置并修改最大worker数
        summary_config = Config.get_summary_llm_config()
        summary_config['max_workers'] = max_workers
        
        # 初始化LLM模型
        self.llm_model = LLMModel(config=summary_config, llm_type="summary")
        
        # 统计变量
        self.success_count = 0
        self.error_count = 0
        self.total_requests = 0
        self.start_time = None
        self.end_time = None
        
        # 线程安全锁
        self.stats_lock = threading.Lock()
        
        # 结果收集
        self.results = []
        self.errors = []
    
    def generate_test_prompts(self, num_prompts: int) -> List[str]:
        """
        生成测试用的提示词
        
        Args:
            num_prompts: 提示词数量
            
        Returns:
            List[str]: 提示词列表
        """
        prompts = []
        
        base_prompts = [
            "请简要介绍人工智能的发展历史。",
            "解释什么是机器学习，并列举3个应用领域。",
            "描述深度学习与传统机器学习的区别。",
            "什么是自然语言处理？给出2个具体例子。",
            "简述计算机视觉在现实生活中的应用。",
            "解释什么是强化学习，并举一个实际应用案例。",
            "描述云计算的主要特点和优势。",
            "什么是大数据？请列举大数据的4V特征。",
            "解释区块链技术的基本原理。",
            "描述物联网的概念及其应用场景。"
        ]
        
        for i in range(num_prompts):
            # 循环使用基础提示词，添加序号使其唯一
            base_prompt = base_prompts[i % len(base_prompts)]
            prompt = f"[请求{i+1}] {base_prompt}"
            prompts.append(prompt)
        
        return prompts
    
    def single_request_test(self, prompt_data: tuple) -> Dict[str, Any]:
        """
        单个请求测试
        
        Args:
            prompt_data: (index, prompt) 元组
            
        Returns:
            Dict: 测试结果
        """
        index, prompt = prompt_data
        request_start = time.time()
        
        try:
            # 调用LLM
            response = self.llm_model.generate_single(prompt)
            request_end = time.time()
            
            # 更新成功统计
            with self.stats_lock:
                self.success_count += 1
            
            result = {
                'index': index,
                'status': 'success',
                'prompt': prompt[:50] + '...',  # 只保存前50字符
                'response_length': len(response),
                'duration': request_end - request_start,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.debug(f"[测试] 请求 {index} 成功，耗时 {result['duration']:.2f}秒")
            return result
            
        except Exception as e:
            request_end = time.time()
            
            # 更新错误统计
            with self.stats_lock:
                self.error_count += 1
            
            error_result = {
                'index': index,
                'status': 'error',
                'prompt': prompt[:50] + '...',
                'error': str(e),
                'duration': request_end - request_start,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.error(f"[测试] 请求 {index} 失败: {e}")
            return error_result
    
    def run_concurrent_test(self, num_requests: int) -> Dict[str, Any]:
        """
        运行并发测试
        
        Args:
            num_requests: 请求总数
            
        Returns:
            Dict: 测试统计结果
        """
        self.logger.info(f"[并发测试] 开始测试：{num_requests} 个请求，{self.max_workers} 个并发worker")
        
        # 生成测试提示词
        prompts = self.generate_test_prompts(num_requests)
        self.total_requests = len(prompts)
        
        # 重置统计
        self.success_count = 0
        self.error_count = 0
        self.results = []
        self.errors = []
        
        # 开始计时
        self.start_time = time.time()
        
        # 使用线程池执行并发请求
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(self.single_request_test, (i, prompt)): i
                for i, prompt in enumerate(prompts)
            }
            
            # 收集结果
            completed = 0
            for future in as_completed(future_to_index):
                result = future.result()
                
                if result['status'] == 'success':
                    self.results.append(result)
                else:
                    self.errors.append(result)
                
                completed += 1
                
                # 每10个请求报告一次进度
                if completed % 10 == 0 or completed == len(prompts):
                    self.logger.info(f"[并发测试] 进度: {completed}/{len(prompts)} ({completed/len(prompts)*100:.1f}%)")
        
        # 结束计时
        self.end_time = time.time()
        
        # 计算统计信息
        total_duration = self.end_time - self.start_time
        success_rate = (self.success_count / self.total_requests) * 100 if self.total_requests > 0 else 0
        qps = self.total_requests / total_duration if total_duration > 0 else 0
        
        # 计算响应时间统计
        success_durations = [r['duration'] for r in self.results]
        avg_response_time = sum(success_durations) / len(success_durations) if success_durations else 0
        max_response_time = max(success_durations) if success_durations else 0
        min_response_time = min(success_durations) if success_durations else 0
        
        stats = {
            'test_config': {
                'total_requests': self.total_requests,
                'max_workers': self.max_workers,
                'total_duration': total_duration
            },
            'results': {
                'success_count': self.success_count,
                'error_count': self.error_count,
                'success_rate': success_rate,
                'qps': qps
            },
            'performance': {
                'avg_response_time': avg_response_time,
                'max_response_time': max_response_time,
                'min_response_time': min_response_time
            },
            'errors': self.errors
        }
        
        return stats
    
    def save_test_results(self, stats: Dict[str, Any], filename: str = None):
        """
        保存测试结果
        
        Args:
            stats: 测试统计结果
            filename: 保存文件名
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"concurrent_test_results_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            self.logger.info(f"[保存结果] 测试结果已保存到: {filename}")
        except Exception as e:
            self.logger.error(f"[保存结果] 保存失败: {e}")
    
    def print_summary(self, stats: Dict[str, Any]):
        """
        打印测试摘要
        
        Args:
            stats: 测试统计结果
        """
        print("\n" + "="*80)
        print("并发LLM测试结果摘要")
        print("="*80)
        
        config = stats['test_config']
        results = stats['results']
        performance = stats['performance']
        
        print(f"测试配置:")
        print(f"  总请求数: {config['total_requests']}")
        print(f"  并发worker数: {config['max_workers']}")
        print(f"  总耗时: {config['total_duration']:.2f} 秒")
        
        print(f"\n结果统计:")
        print(f"  成功请求: {results['success_count']}")
        print(f"  失败请求: {results['error_count']}")
        print(f"  成功率: {results['success_rate']:.2f}%")
        print(f"  QPS (每秒请求数): {results['qps']:.2f}")
        
        print(f"\n性能统计:")
        print(f"  平均响应时间: {performance['avg_response_time']:.2f} 秒")
        print(f"  最大响应时间: {performance['max_response_time']:.2f} 秒")
        print(f"  最小响应时间: {performance['min_response_time']:.2f} 秒")
        
        if stats['errors']:
            print(f"\n错误详情 (显示前5个):")
            for i, error in enumerate(stats['errors'][:5]):
                print(f"  {i+1}. 请求{error['index']}: {error['error']}")
        
        print("="*80)


def main():
    """主函数"""
    print("开始200并发worker LLM请求测试...")
    
    # 测试参数
    MAX_WORKERS = 200
    NUM_REQUESTS = 2000  # 可以根据需要调整请求数量
    
    # 创建测试器
    tester = ConcurrentLLMTester(max_workers=MAX_WORKERS)
    
    try:
        # 运行测试
        stats = tester.run_concurrent_test(NUM_REQUESTS)
        
        # 打印摘要
        tester.print_summary(stats)
        
        # 保存结果
        tester.save_test_results(stats)
        
        # 检查是否有错误
        if stats['results']['error_count'] > 0:
            print(f"\n⚠️  警告: {stats['results']['error_count']} 个请求失败")
            return 1
        else:
            
            print(f"\n✅ 测试完成: 所有 {stats['results']['success_count']} 个请求都成功!")
            return 0
            
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)