#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立测试embedding模型批量处理能力的脚本
测试不同批量大小下的处理性能和准确性
"""

import time
import numpy as np
from typing import List, Dict, Any
import requests
import json
# from sentence_transformers import SentenceTransformer
# import torch


class EmbeddingBatchTester:
    """embedding模型批量处理能力测试器"""
    
    def __init__(self):
        # 使用实际的embedding配置
        self.api_url = "http://localhost:8000/v1/embeddings"
        self.api_key = "EMPTY"
        self.model_name = "BAAI/bge-m3"
        self.default_batch_size = 8
        self.max_length = 512
        self.timeout = 180
        self.test_texts = [
            "这是第一个测试文本，用于验证embedding模型的基本功能。",
            "人工智能技术正在快速发展，改变着我们的生活方式。",
            "机器学习算法可以从大量数据中学习模式和规律。",
            "自然语言处理是计算机科学和人工智能的重要分支。",
            "深度学习网络能够处理复杂的非线性关系。",
            "文本向量化是信息检索系统的核心技术之一。",
            "聚类算法可以将相似的数据点分组到同一类别中。",
            "语义相似度计算在问答系统中发挥重要作用。",
            "知识图谱结合了符号推理和统计学习方法。",
            "检索增强生成技术提高了问答系统的准确性。"
        ]
        
        # 扩展测试数据到不同规模
        self.small_batch = self.test_texts[:5]
        self.medium_batch = self.test_texts * 5  # 50个文本
        self.large_batch = self.test_texts * 20  # 200个文本
        
    def test_api_embedding(self, texts: List[str], batch_size: int = None) -> Dict[str, Any]:
        """测试API embedding模型的批量处理"""
        if batch_size is None:
            batch_size = self.default_batch_size
        print(f"测试API embedding - 文本数量: {len(texts)}, 批量大小: {batch_size}")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        start_time = time.time()
        embeddings = []
        
        try:
            # 分批处理
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # 构建符合OpenAI API格式的请求
                payload = {
                    "input": batch_texts,
                    "model": self.model_name,
                    "encoding_format": "float"
                }
                
                response = requests.post(
                    self.api_url, 
                    json=payload, 
                    headers=headers, 
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # OpenAI API格式的返回结果
                    if "data" in result:
                        batch_embeddings = [item["embedding"] for item in result["data"]]
                        embeddings.extend(batch_embeddings)
                    else:
                        print(f"API返回格式错误: {result}")
                        return {"success": False, "error": "API返回格式错误"}
                else:
                    print(f"批次 {i//batch_size + 1} API调用失败: {response.status_code}")
                    print(f"响应内容: {response.text}")
                    return {"success": False, "error": f"HTTP {response.status_code}"}
                        
        except Exception as e:
            print(f"API调用异常: {str(e)}")
            return {"success": False, "error": str(e)}
        
        end_time = time.time()
        
        return {
            "success": True,
            "method": "API",
            "text_count": len(texts),
            "batch_size": batch_size,
            "embedding_dim": len(embeddings[0]) if embeddings else 0,
            "processing_time": end_time - start_time,
            "throughput": len(texts) / (end_time - start_time),
            "embeddings": embeddings
        }
    
    def test_local_embedding(self, texts: List[str], batch_size: int = None) -> Dict[str, Any]:
        """测试本地embedding模型的批量处理"""
        print(f"测试本地embedding - 文本数量: {len(texts)}, 批量大小: {batch_size or '默认'}")
        
        try:
            # 加载本地模型
            model = SentenceTransformer('all-MiniLM-L6-v2')
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model.to(device)
            
            start_time = time.time()
            
            if batch_size is None:
                # 一次性处理所有文本
                embeddings = model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
            else:
                # 分批处理
                embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_embeddings = model.encode(batch_texts, convert_to_tensor=False)
                    embeddings.extend(batch_embeddings)
                embeddings = np.array(embeddings)
            
            end_time = time.time()
            
            return {
                "success": True,
                "method": "Local",
                "text_count": len(texts),
                "batch_size": batch_size,
                "embedding_dim": embeddings.shape[1],
                "processing_time": end_time - start_time,
                "throughput": len(texts) / (end_time - start_time),
                "device": device,
                "embeddings": embeddings.tolist()
            }
            
        except Exception as e:
            print(f"本地模型处理异常: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def compare_batch_sizes(self, texts: List[str], batch_sizes: List[int], method: str = "api"):
        """比较不同批量大小的处理性能"""
        print(f"\n=== 批量大小性能比较 ({method}模型) ===")
        print(f"测试文本数量: {len(texts)}")
        
        results = []
        
        for batch_size in batch_sizes:
            print(f"\n测试批量大小: {batch_size}")
            
            if method == "local":
                result = self.test_local_embedding(texts, batch_size)
            elif method == "api":
                result = self.test_api_embedding(texts, batch_size)
            else:
                print(f"未知的测试方法: {method}")
                continue
            
            if result["success"]:
                results.append(result)
                print(f"处理时间: {result['processing_time']:.2f}秒")
                print(f"吞吐量: {result['throughput']:.2f}文本/秒")
            else:
                print(f"处理失败: {result.get('error', '未知错误')}")
        
        # 分析最优批量大小
        if results:
            best_result = max(results, key=lambda x: x['throughput'])
            print(f"\n最优批量大小: {best_result['batch_size']}")
            print(f"最高吞吐量: {best_result['throughput']:.2f}文本/秒")
        
        return results
    
    def test_embedding_consistency(self, texts: List[str]):
        """测试批量处理和单个处理的一致性"""
        print(f"\n=== 批量处理一致性测试 ===")
        
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # 批量处理
            batch_embeddings = model.encode(texts, convert_to_tensor=False)
            
            # 单个处理
            single_embeddings = []
            for text in texts:
                embedding = model.encode([text], convert_to_tensor=False)[0]
                single_embeddings.append(embedding)
            single_embeddings = np.array(single_embeddings)
            
            # 计算相似度
            similarities = []
            for i in range(len(texts)):
                similarity = np.dot(batch_embeddings[i], single_embeddings[i]) / (
                    np.linalg.norm(batch_embeddings[i]) * np.linalg.norm(single_embeddings[i])
                )
                similarities.append(similarity)
            
            avg_similarity = np.mean(similarities)
            min_similarity = np.min(similarities)
            
            print(f"平均余弦相似度: {avg_similarity:.6f}")
            print(f"最小余弦相似度: {min_similarity:.6f}")
            
            if avg_similarity > 0.999:
                print("✅ 批量处理和单个处理结果高度一致")
            elif avg_similarity > 0.99:
                print("⚠️  批量处理和单个处理结果基本一致")
            else:
                print("❌ 批量处理和单个处理结果存在显著差异")
                
            return {
                "avg_similarity": avg_similarity,
                "min_similarity": min_similarity,
                "consistent": avg_similarity > 0.999
            }
            
        except Exception as e:
            print(f"一致性测试失败: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def test_api_connection(self):
        """测试API连接和基本功能"""
        print("=" * 60)
        print("API连接测试")
        print("=" * 60)
        
        test_text = ["这是一个简单的测试文本。"]
        result = self.test_api_embedding(test_text, batch_size=1)
        
        if result["success"]:
            print("✅ API连接成功")
            print(f"模型: {self.model_name}")
            print(f"向量维度: {result['embedding_dim']}")
            print(f"处理时间: {result['processing_time']:.3f}秒")
            return True
        else:
            print("❌ API连接失败")
            print(f"错误: {result.get('error', '未知错误')}")
            return False
    
    def run_comprehensive_test(self):
        """运行综合性能测试"""
        print("=" * 60)
        print("Embedding模型批量处理能力综合测试")
        print("=" * 60)
        
        # 测试1: API模型不同规模数据的处理性能
        print("\n1. API模型不同规模数据处理性能测试")
        test_cases = [
            ("小批量", self.small_batch),
            ("中批量", self.medium_batch),
            ("大批量", self.large_batch)
        ]
        
        for name, texts in test_cases:
            print(f"\n--- {name}数据测试 ({len(texts)}个文本) ---")
            api_result = self.test_api_embedding(texts)
            if api_result["success"]:
                print(f"API模型处理时间: {api_result['processing_time']:.2f}秒")
                print(f"API模型吞吐量: {api_result['throughput']:.2f}文本/秒")
                print(f"向量维度: {api_result['embedding_dim']}")
            else:
                print(f"API测试失败: {api_result.get('error', '未知错误')}")
        
        # 测试2: 不同批量大小性能比较
        print("\n2. API模型批量大小优化测试")
        batch_sizes = [1, 4, 8, 16, 32]
        self.compare_batch_sizes(self.medium_batch, batch_sizes, "api")
        
        # 测试3: 默认批量大小性能测试
        print(f"\n3. 默认批量大小({self.default_batch_size})性能测试")
        default_result = self.test_api_embedding(self.medium_batch)
        if default_result["success"]:
            print(f"处理时间: {default_result['processing_time']:.2f}秒")
            print(f"吞吐量: {default_result['throughput']:.2f}文本/秒")
        
        print("\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)


def main():
    """主函数"""
    import sys
    
    tester = EmbeddingBatchTester()
    
    # 如果有命令行参数，则运行特定测试
    if len(sys.argv) > 1:
        if sys.argv[1] == "--connection":
            tester.test_api_connection()
            return
        elif sys.argv[1] == "--quick":
            print("快速批量测试")
            result = tester.test_api_embedding(tester.small_batch)
            if result["success"]:
                print(f"处理{len(tester.small_batch)}个文本")
                print(f"处理时间: {result['processing_time']:.2f}秒")
                print(f"吞吐量: {result['throughput']:.2f}文本/秒")
            return
    
    # 默认运行完整测试
    # 首先测试连接
    if not tester.test_api_connection():
        print("\n⚠️ API连接失败，跳过后续测试")
        return
    
    # 连接成功后运行完整测试
    tester.run_comprehensive_test()


if __name__ == "__main__":
    main()