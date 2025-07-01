#!/usr/bin/env python3
"""
CS JSONL转换器
将CS JSONL格式的数据转换为与Musique数据集相同的格式
对过长的context按句子分片，每个分片保持在100-150 tokens
"""

import json
import re
import os
from typing import List
import argparse

def count_tokens(text: str) -> int:
    """粗略估算token数量，按空格分词"""
    return len(text.split())

def split_text_by_sentences(text: str, min_tokens: int = 300, max_tokens: int = 500) -> List[str]:
    """
    按句子分片文本，每个分片保持在min_tokens到max_tokens之间
    """
    # 按句号、问号、感叹号分句
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # 如果当前句子加上当前块会超过最大token数
        potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
        potential_tokens = count_tokens(potential_chunk)
        
        if potential_tokens > max_tokens and current_chunk:
            # 如果当前块已经达到最小token要求，保存它
            if count_tokens(current_chunk) >= min_tokens:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # 否则继续添加，即使会超过最大限制
                current_chunk = potential_chunk
        else:
            current_chunk = potential_chunk
    
    # 添加最后一个块
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def convert_csjsonl_to_musique(input_file: str, output_dir: str):
    """
    将CS JSONL格式转换为Musique格式，生成corpus和questions文件
    """
    converted_items = []
    questions = []
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                
                # 提取基本字段
                question_text = data.get('input', '')
                answers = data.get('answers', [])
                context = data.get('context', '')
                
                if not context.strip() or not question_text.strip():
                    continue
                
                # 创建问题条目
                question_item = {
                    "id": f"cs_question_{line_num + 1}",
                    "question": question_text,
                    "answers": answers,
                    "type": "bridge"  # 默认类型
                }
                questions.append(question_item)
                
                # 检查context长度，如果超过150 tokens则分片
                context_tokens = count_tokens(context)
                
                if context_tokens <= 150:
                    # 不需要分片
                    item = {
                        "idx": len(converted_items),
                        "title": f"CS Document {line_num + 1}",
                        "context": context.strip(),
                        "is_supporting": True,
                        "corresponding_question_md5": f"cs_question_{line_num + 1}",
                        "id": len(converted_items)
                    }
                    converted_items.append(item)
                else:
                    # 需要分片
                    chunks = split_text_by_sentences(context)
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        item = {
                            "idx": len(converted_items),
                            "title": f"CS Document {line_num + 1} - Part {chunk_idx + 1}",
                            "context": chunk,
                            "is_supporting": True if chunk_idx == 0 else False,  # 只有第一个分片标记为supporting
                            "corresponding_question_md5": f"cs_question_{line_num + 1}",
                            "id": len(converted_items)
                        }
                        converted_items.append(item)
                        
            except json.JSONDecodeError as e:
                print(f"解析第{line_num + 1}行时出错: {e}")
                continue
            except Exception as e:
                print(f"处理第{line_num + 1}行时出错: {e}")
                continue
    
    # 写入corpus文件
    corpus_file = os.path.join(output_dir, 'corpus.json')
    with open(corpus_file, 'w', encoding='utf-8') as f:
        for item in converted_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 写入questions文件
    questions_file = os.path.join(output_dir, 'questions.json') 
    with open(questions_file, 'w', encoding='utf-8') as f:
        for question in questions:
            f.write(json.dumps(question, ensure_ascii=False) + '\n')
    
    print(f"转换完成！")
    print(f"原始条目数: {line_num + 1}")
    print(f"转换后corpus条目数: {len(converted_items)}")
    print(f"生成问题数: {len(questions)}")
    print(f"输出文件:")
    print(f"  - {corpus_file}")
    print(f"  - {questions_file}")

def main():
    parser = argparse.ArgumentParser(description='将CS JSONL格式转换为Musique格式')
    parser.add_argument('input_file', help='输入的CS JSONL文件路径')
    parser.add_argument('output_dir', help='输出目录路径')
    
    args = parser.parse_args()
    
    convert_csjsonl_to_musique(args.input_file, args.output_dir)

if __name__ == "__main__":
    main()