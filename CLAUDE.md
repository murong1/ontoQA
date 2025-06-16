# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述
这是一个研究项目，用于研究通过聚类相似文本片段，使用LLM总结聚类的本体，最终进行RAG操作来增强模型能力。项目使用Python开发。

**重要提醒：**
- 这是研究项目，不是在线产品开发，不要生成web内容
- 不要自行添加功能或优化操作，只完成指定任务
- 保证实验的完整性和最小性，不需要考虑前一版本兼容，直接修改不兼容的代码
- 没有直接要求，不要写兜底逻辑
- 所有注释和对话使用中文，代码使用英文
- 代码修改应确保整体性，测试时使用最新代码

## 数据集
- 示例语料库：`datasets/musique/test_corpus_100.json`
- 问题集：`datasets/musique/test_questions_20.json`

## 常用命令

### 运行实验
```bash
# 运行完整的OntoQA实验（聚类+本体总结+RAG）
python main.py

# 运行zero-shot基线实验（直接问答，无检索）
python main.py --zeroshot

# 运行消融实验（仅使用原始文档检索，不包含本体总结）
python main.py --ablation

# 自定义参数运行
python main.py --corpus datasets/musique/test_corpus_100.json --questions datasets/musique/test_questions_20.json --clusters 5 --output results

# 启用详细日志
python main.py --verbose
```

### 环境设置
```bash
# 安装依赖
pip install -r requirements.txt
```

## 系统架构

### 核心模块
1. **DataLoader** (`src/data_loader.py`) - 数据加载，处理语料库和问题数据
2. **TextClusterer** (`src/clustering.py`) - 文本聚类，使用KMeans对文档进行聚类（3-10个聚类）
3. **OntologySummarizer** (`src/summarizer.py`) - 本体总结，使用LLM为每个聚类生成本体摘要
4. **RAGSystem** (`src/rag_system.py`) - RAG系统，使用FAISS构建向量索引并执行检索增强生成
5. **Evaluator** (`src/evaluator.py`) - 评估器，评估问答性能并生成结果报告

### 支持模块
- **EmbeddingModel** (`src/embedding_model.py`) - 文本嵌入，支持API和本地模型
- **LLMModel** (`src/llm_model.py`) - 大语言模型接口，支持OpenAI API
- **Config** (`config.py`) - 配置管理，包含LLM、嵌入模型等所有配置

### 实验流程
1. **数据加载** - 加载语料库文档和测试问题
2. **文本聚类** - 使用嵌入模型计算文档向量，KMeans聚类
3. **本体总结** - 为每个聚类生成本体摘要描述
4. **RAG索引构建** - 结合原始文档和本体摘要构建FAISS向量索引
5. **问答评估** - 对测试问题进行检索增强问答并评估性能

### 缓存机制
- FAISS索引自动缓存在 `cache/indexes/` 目录
- 缓存基于语料库文件路径和聚类数量的哈希值
- 相同参数的实验会自动使用缓存的索引

### 配置要点
- LLM配置在 `Config` 类中，支持问答和总结使用不同模型
- 嵌入模型支持API调用和本地模型
- 聚类数量限制为3-10个
- 所有结果保存在 `results/` 目录，包含详细评估报告
