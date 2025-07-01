#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较并发优化前后的性能差异
"""

import time
import logging
from typing import Dict, List, Any
from src.summarizer import OntologySummarizer

# 设置日志级别以减少输出
logging.basicConfig(level=logging.WARNING)

def create_test_data(num_clusters: int = 8, docs_per_cluster: int = 4) -> Dict[int, List[Dict[str, Any]]]:
    """创建测试数据"""
    test_clusters = {}
    
    topics = [
        "机器学习算法", "深度神经网络", "自然语言处理技术", "计算机视觉系统",
        "数据挖掘方法", "推荐系统算法", "语音识别技术", "图像处理算法",
        "强化学习模型", "知识图谱构建", "信息检索系统", "文本分类技术"
    ]
    
    for i in range(num_clusters):
        topic = topics[i % len(topics)]
        documents = []
        
        for j in range(docs_per_cluster):
            doc = {
                'id': f'doc_{i}_{j}',
                'title': f'{topic}研究文档{j+1}',
                'context': f'这是关于{topic}的详细技术文档。{topic}是人工智能和机器学习领域的核心技术，涉及复杂的算法设计、数据结构优化、模型训练策略、性能评估方法等多个技术层面。该技术在实际工程应用中具有重要价值，包括但不限于自动化系统、智能推荐、决策支持、模式识别等应用场景。技术实现需要考虑计算复杂度、内存使用、并发处理、错误处理等工程因素。'
            }
            documents.append(doc)
        
        test_clusters[i] = documents
    
    return test_clusters

def test_concurrent_performance():
    """测试并发版本性能"""
    print("🚀 测试并发优化版本...")
    
    test_data = create_test_data(num_clusters=8, docs_per_cluster=3)
    
    # 使用并发优化的配置
    summarizer = OntologySummarizer(
        output_dir="performance_results",
        max_concurrent_llm=5,     # 并发LLM调用
        max_concurrent_embedding=3  # 并发嵌入计算
    )
    
    start_time = time.time()
    result = summarizer.summarize_clusters(test_data)
    end_time = time.time()
    
    processing_time = end_time - start_time
    
    return {
        'version': 'concurrent',
        'clusters': len(test_data),
        'total_docs': sum(len(docs) for docs in test_data.values()),
        'output_ontologies': len(result),
        'processing_time': processing_time,
        'avg_time_per_cluster': processing_time / len(test_data)
    }

def test_sequential_simulation():
    """模拟顺序处理版本（通过限制并发数为1）"""
    print("🐌 测试顺序处理版本...")
    
    test_data = create_test_data(num_clusters=8, docs_per_cluster=3)
    
    # 限制并发数为1来模拟顺序处理
    summarizer = OntologySummarizer(
        output_dir="performance_results",
        max_concurrent_llm=1,      # 顺序LLM调用
        max_concurrent_embedding=1   # 顺序嵌入计算
    )
    
    start_time = time.time()
    result = summarizer.summarize_clusters(test_data)
    end_time = time.time()
    
    processing_time = end_time - start_time
    
    return {
        'version': 'sequential',
        'clusters': len(test_data),
        'total_docs': sum(len(docs) for docs in test_data.values()),
        'output_ontologies': len(result),
        'processing_time': processing_time,
        'avg_time_per_cluster': processing_time / len(test_data)
    }

def compare_performance():
    """比较性能"""
    print("=== 本体总结器并发性能比较测试 ===\n")
    
    # 测试顺序处理版本
    sequential_result = test_sequential_simulation()
    
    print(f"顺序处理完成: {sequential_result['processing_time']:.2f} 秒\n")
    
    # 测试并发处理版本
    concurrent_result = test_concurrent_performance()
    
    print(f"并发处理完成: {concurrent_result['processing_time']:.2f} 秒\n")
    
    # 计算性能提升
    speedup = sequential_result['processing_time'] / concurrent_result['processing_time']
    time_saved = sequential_result['processing_time'] - concurrent_result['processing_time']
    efficiency_gain = ((sequential_result['processing_time'] - concurrent_result['processing_time']) / sequential_result['processing_time']) * 100
    
    # 输出详细比较结果
    print("=" * 60)
    print("📊 性能比较结果")
    print("=" * 60)
    
    print(f"测试数据规模:")
    print(f"  • 聚类数量: {sequential_result['clusters']}")
    print(f"  • 文档总数: {sequential_result['total_docs']}")
    print(f"  • 输出本体数: {sequential_result['output_ontologies']}")
    print()
    
    print(f"处理时间对比:")
    print(f"  • 顺序处理: {sequential_result['processing_time']:.2f} 秒")
    print(f"  • 并发处理: {concurrent_result['processing_time']:.2f} 秒")
    print(f"  • 节省时间: {time_saved:.2f} 秒")
    print()
    
    print(f"性能指标:")
    print(f"  • 速度提升: {speedup:.2f}x")
    print(f"  • 效率提升: {efficiency_gain:.1f}%")
    print(f"  • 平均每聚类处理时间:")
    print(f"    - 顺序: {sequential_result['avg_time_per_cluster']:.2f} 秒")
    print(f"    - 并发: {concurrent_result['avg_time_per_cluster']:.2f} 秒")
    print()
    
    # 性能评估
    if speedup >= 2.0:
        grade = "🏆 优秀"
    elif speedup >= 1.5:
        grade = "🥈 良好"
    elif speedup >= 1.2:
        grade = "🥉 一般"
    else:
        grade = "⚠️  需要优化"
    
    print(f"并发优化评级: {grade}")
    print(f"建议: ", end="")
    
    if speedup >= 2.0:
        print("并发优化效果显著，建议在生产环境中使用。")
    elif speedup >= 1.5:
        print("并发优化效果良好，可以投入使用。")
    elif speedup >= 1.2:
        print("有一定改善，可考虑进一步优化并发策略。")
    else:
        print("并发效果不明显，建议检查瓶颈或调整并发参数。")
    
    print("=" * 60)
    
    return {
        'sequential': sequential_result,
        'concurrent': concurrent_result,
        'speedup': speedup,
        'efficiency_gain': efficiency_gain
    }

if __name__ == "__main__":
    try:
        results = compare_performance()
        print(f"\n✅ 性能比较测试完成！")
        print(f"并发优化实现了 {results['speedup']:.2f}x 的性能提升")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()