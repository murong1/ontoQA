#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OWL转换器模块
将JSON格式的本体数据转换为标准的OWL格式
支持Turtle（用于查看调试）和OWL/XML（用于标准兼容）两种输出格式
"""

import json
import logging
import re
from typing import Dict, List, Any, Set, Tuple, TypedDict
from datetime import datetime
from pathlib import Path


class JSONToOWLConverter:
    """JSON本体到OWL格式转换器"""
    
    def __init__(self, namespace_uri: str = "http://ontoqa.org/ontology#"):
        """
        初始化转换器
        
        Args:
            namespace_uri: 本体命名空间URI
        """
        # 使用模块特定的logger而不是根logger
        self.logger = logging.getLogger("json2owl")
        self.namespace_uri = namespace_uri
        self.namespace_prefix = "ontoqa"
        
        # 预定义的命名空间
        self.namespaces = {
            "owl": "http://www.w3.org/2002/07/owl#",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            self.namespace_prefix: namespace_uri
        }
        
        # 移除关系映射，直接使用驼峰转换
        
        # 用于缓存已清理的名称，确保同一概念映射到同一类
        self._name_cache = {}
    
    def convert_json_to_owl(self, json_file_path: str, output_dir: str = "results/owl", 
                           formats: List[str] = None) -> Dict[str, str]:
        """
        将JSON本体转换为OWL格式
        
        Args:
            json_file_path: JSON本体文件路径
            output_dir: 输出目录
            formats: 输出格式列表 ['turtle', 'xml', 'all']，默认生成所有格式
            
        Returns:
            Dict: 包含生成文件路径的字典 {"turtle": path, "owl_xml": path}
            
        Raises:
            FileNotFoundError: 当JSON文件不存在时
            json.JSONDecodeError: 当JSON文件格式错误时
            PermissionError: 当没有文件写入权限时
        """
        try:
            self.logger.info(f"开始转换JSON本体到OWL格式: {json_file_path}")
            
            # 使用pathlib处理路径
            json_path = Path(json_file_path)
            output_path = Path(output_dir)
            
            # 检查输入文件是否存在
            if not json_path.exists():
                raise FileNotFoundError(f"JSON文件不存在: {json_file_path}")
            
            # 确保输出目录存在
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 加载JSON数据
            with json_path.open('r', encoding='utf-8') as f:
                json_data = json.load(f)
                
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON文件格式错误: {e}")
            raise
        except FileNotFoundError as e:
            self.logger.error(f"文件不存在: {e}")
            raise
        except PermissionError as e:
            self.logger.error(f"文件权限错误: {e}")
            raise
        except Exception as e:
            self.logger.error(f"转换过程中出现错误: {e}")
            raise
        
        try:
            # 解析JSON数据
            ontologies = json_data.get('ontologies', [])
            metadata = json_data.get('metadata', {})
            
            self.logger.info(f"解析到 {len(ontologies)} 个本体概念")
            
            # 提取类和属性信息
            classes, properties, object_relations, subclass_relations = self._extract_owl_elements(ontologies)
            
            # 设置默认格式
            if formats is None:
                formats = ['turtle', 'xml']
            elif 'all' in formats:
                formats = ['turtle', 'xml']
            
            # 生成简洁的文件名
            result = {}
            
            # 按需求生成文件
            if 'turtle' in formats:
                turtle_file = output_path / "ontology.ttl"
                self._generate_turtle(classes, properties, object_relations, subclass_relations, str(turtle_file), metadata)
                result['turtle'] = str(turtle_file)
                self.logger.info(f"  Turtle格式: {turtle_file}")
            
            if 'xml' in formats:
                owl_xml_file = output_path / "ontology.owl"
                self._generate_owl_xml(classes, properties, object_relations, subclass_relations, str(owl_xml_file), metadata)
                result['owl_xml'] = str(owl_xml_file)
                self.logger.info(f"  OWL/XML格式: {owl_xml_file}")
            
            self.logger.info(f"OWL转换完成，生成 {len(result)} 个文件")
            return result
            
        except Exception as e:
            self.logger.error(f"生成OWL文件时出现错误: {e}")
            raise
    
    def _extract_owl_elements(self, ontologies: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Set[str]]], List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        """
        从JSON本体中提取OWL元素
        
        Returns:
            Tuple: (classes, properties, object_relations, subclass_relations)
                - classes: {class_name: {description, source_info}}
                - properties: {property_name: {domains, ranges}}
                - object_relations: [(subject, predicate, object)] - OTHER类型的关系
                - subclass_relations: [(subclass, 'subClassOf', superclass)] - TYPE类型的关系
        """
        classes = {}
        properties = {}
        object_relations = []  # OTHER类型的关系
        subclass_relations = []  # TYPE类型的关系
        
        for ontology in ontologies:
            class_name = self._sanitize_name(ontology['ontology_name'])
            description = ontology.get('description', '')
            
            # 清理描述文本（移除关系部分）
            clean_description = self._clean_description(description)
            
            # 添加类（如果已存在则合并信息）
            if class_name in classes:
                # 合并文档数量
                classes[class_name]['document_count'] += ontology.get('document_count', 0)
                # 合并描述（如果新描述更详细）
                if len(clean_description) > len(classes[class_name]['description']):
                    classes[class_name]['description'] = clean_description
            else:
                classes[class_name] = {
                    'description': clean_description,
                    'original_name': ontology['ontology_name'],
                    'document_count': ontology.get('document_count', 0),
                    'cluster_id': ontology.get('cluster_id', -1)
                }
            
            # 处理关系
            for relationship in ontology.get('relationships', []):
                subject, predicate, obj, rel_type = self._parse_relationship(relationship)
                if subject and predicate and obj:
                    # 根据关系类型分类
                    if rel_type == 'TYPE':
                        subclass_relations.append((subject, predicate, obj))
                    else:  # OTHER类型
                        object_relations.append((subject, predicate, obj))
                        
                        # 只为OTHER类型的关系收集属性信息
                        if predicate not in properties:
                            properties[predicate] = {'domains': set(), 'ranges': set()}
                        
                        properties[predicate]['domains'].add(subject)
                        properties[predicate]['ranges'].add(obj)
                        
            # 处理关系中提到但未定义的类
            for relationship in ontology.get('relationships', []):
                subject, predicate, obj, rel_type = self._parse_relationship(relationship)
                if subject and predicate and obj:
                    # 确保所有关系中的类都被定义
                    for class_name in [subject, obj]:
                        if class_name not in classes:
                            classes[class_name] = {
                                'description': f"{class_name}类（从关系中推断）",
                                'original_name': class_name,
                                'document_count': 0,
                                'cluster_id': -999
                            }
        
        self.logger.info(f"提取到 {len(classes)} 个类, {len(properties)} 个对象属性, {len(object_relations)} 个对象关系, {len(subclass_relations)} 个子类关系")
        
        return classes, properties, object_relations, subclass_relations
    
    def _to_camel_case(self, text: str) -> str:
        """
        将文本转换为驼峰命名（首字母小写）
        例如: "is type of" -> "isTypeOf"
             "influences" -> "influences"
             "regulated by" -> "regulatedBy"
        """
        if not text or not text.strip():
            return "unknownRelation"
        
        # 移除特殊字符，保留字母数字和空格
        cleaned = re.sub(r'[^\w\s]', '', text.strip())
        
        # 按空格分词
        words = cleaned.split()
        
        if not words:
            return "unknownRelation"
        
        # 第一个单词小写，后续单词首字母大写
        camel_case = words[0].lower()
        for word in words[1:]:
            if word:  # 跳过空字符串
                camel_case += word.capitalize()
        
        return camel_case
    
    def _sanitize_name(self, name: str) -> str:
        """
        清理名称，转换为合法的OWL标识符，确保同一概念映射到同一类
        """
        if not name or not name.strip():
            return "UnknownClass"
        
        # 使用原始名称作为缓存键，确保相同概念得到相同的清理名称
        original_name = name.strip()
        if original_name in self._name_cache:
            return self._name_cache[original_name]
            
        # 移除特殊字符，保留字母数字、空格和连字符
        sanitized = re.sub(r'[^\w\s-]', '', original_name)
        # 按空格和连字符分词
        words = re.split(r'[\s\-]+', sanitized)
        # 过滤空字符串
        words = [word for word in words if word.strip()]
        if not words:
            cleaned_name = "UnknownClass"
        else:
            camel_case = words[0].capitalize()
            for word in words[1:]:
                camel_case += word.capitalize()
            cleaned_name = camel_case
        
        # 缓存结果
        self._name_cache[original_name] = cleaned_name
        return cleaned_name
    
    def _clean_description(self, description: str) -> str:
        """
        清理描述文本，移除关系部分
        """
        # 使用正则表达式处理各种关系分隔符
        # 支持中英文、大小写变体、不同标点符号
        pattern = r'(关系\s*[:：]|[Rr]elations?\s*:|[Rr]elationships?\s*:)'
        parts = re.split(pattern, description, 1)
        return parts[0].strip() if parts else description.strip()
    
    def _parse_relationship(self, relationship_str: str) -> Tuple[str, str, str, str]:
        """
        解析关系字符串，格式为: "{'relation': 'Subject -> Predicate -> Object', 'type': 'TYPE/OTHER'}"
        
        Returns:
            Tuple: (subject, predicate, object, relation_type) 或 (None, None, None, None) 如果解析失败
        """
        try:
            # 使用ast.literal_eval安全地解析字符串形式的字典
            import ast
            rel_dict = ast.literal_eval(relationship_str)
            
            if not isinstance(rel_dict, dict) or 'relation' not in rel_dict or 'type' not in rel_dict:
                self.logger.warning(f"关系格式错误: {relationship_str}")
                return None, None, None, None
                
            relation = rel_dict['relation']
            relation_type = rel_dict['type']
            
            # 解析关系字符串 "Subject -> Predicate -> Object"
            parts = re.split(r'\s*->\s*', relation)
            if len(parts) != 3:
                self.logger.warning(f"无法解析关系: {relation}")
                return None, None, None, None
            
            subject = self._sanitize_name(parts[0].strip())
            predicate_raw = parts[1].strip()
            obj = self._sanitize_name(parts[2].strip())
            
            # 对于TYPE类型，谓词固定为subClassOf
            if relation_type == 'TYPE':
                predicate = 'subClassOf'
            else:
                # 其他关系转换为驼峰命名
                predicate = self._to_camel_case(predicate_raw)
            
            return subject, predicate, obj, relation_type
            
        except Exception as e:
            self.logger.warning(f"解析关系时出错: {relationship_str}, 错误: {e}")
            return None, None, None, None
    
    def _generate_turtle(self, classes: Dict[str, Dict], properties: Dict[str, Set], 
                        object_relations: List[Tuple[str, str, str]], subclass_relations: List[Tuple[str, str, str]], 
                        output_file: str, metadata: Dict):
        """
        生成Turtle格式文件
        """
        self.logger.info(f"生成Turtle格式: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # 写入前缀声明
            f.write("# OntoQA 本体 - Turtle格式\n")
            f.write(f"# 生成时间: {datetime.now().isoformat()}\n")
            f.write(f"# 源数据: {metadata.get('corpus_file', 'unknown')}\n")
            f.write(f"# 文档数量: {metadata.get('document_count', 0)}\n")
            f.write(f"# 聚类数量: {metadata.get('cluster_count', 0)}\n")
            f.write(f"# 本体数量: {metadata.get('ontology_count', 0)}\n\n")
            
            for prefix, uri in self.namespaces.items():
                f.write(f"@prefix {prefix}: <{uri}> .\n")
            f.write("\n")
            
            # 本体声明
            f.write("# 本体声明\n")
            f.write(f"<{self.namespace_uri[:-1]}> rdf:type owl:Ontology ;\n")
            f.write(f'    rdfs:label "OntoQA Research Ontology" ;\n')
            f.write(f'    rdfs:comment "从文档聚类中提取的领域本体，用于OntoQA研究项目" .\n\n')
            
            # 类定义
            f.write("# 类定义\n")
            for class_name, class_info in classes.items():
                f.write(f"{self.namespace_prefix}:{class_name} rdf:type owl:Class ;\n")
                f.write(f'    rdfs:label "{class_info["original_name"]}" ;\n')
                if class_info['description']:
                    escaped_desc = class_info['description'].replace('"', '\\"').replace('\n', '\\n')
                    f.write(f'    rdfs:comment "{escaped_desc}" ;\n')
                f.write(f'    ontoqa:documentCount {class_info["document_count"]} ;\n')
                f.write(f'    ontoqa:clusterId {class_info["cluster_id"]} .\n\n')
            
            # 数据属性定义
            f.write("# 数据属性定义\n")
            data_properties = [
                ("documentCount", "文档数量", "xsd:integer"),
                ("clusterId", "聚类ID", "xsd:integer"),
                ("generatedFrom", "生成源文件", "xsd:string"),
                ("clusterCount", "聚类数量", "xsd:integer"),
                ("ontologyCount", "本体数量", "xsd:integer")
            ]
            
            for prop_name, prop_label, prop_range in data_properties:
                f.write(f"{self.namespace_prefix}:{prop_name} rdf:type owl:DatatypeProperty ;\n")
                f.write(f'    rdfs:label "{prop_label}" ;\n')
                f.write(f"    rdfs:range {prop_range} .\n\n")
            
            # 对象属性定义
            f.write("# 对象属性定义\n")
            for prop_name, prop_info in properties.items():
                f.write(f"{self.namespace_prefix}:{prop_name} rdf:type owl:ObjectProperty ;\n")
                f.write(f'    rdfs:label "{prop_name}" ;\n')
                
                # 定义域
                if len(prop_info['domains']) == 1:
                    domain = list(prop_info['domains'])[0]
                    f.write(f"    rdfs:domain {self.namespace_prefix}:{domain} ;\n")
                elif len(prop_info['domains']) > 1:
                    domains = " ".join([f"{self.namespace_prefix}:{d}" for d in prop_info['domains']])
                    f.write(f"    rdfs:domain [ rdf:type owl:Class ; owl:unionOf ( {domains} ) ] ;\n")
                
                # 值域
                if len(prop_info['ranges']) == 1:
                    range_class = list(prop_info['ranges'])[0]
                    f.write(f"    rdfs:range {self.namespace_prefix}:{range_class} .\n\n")
                elif len(prop_info['ranges']) > 1:
                    ranges = " ".join([f"{self.namespace_prefix}:{r}" for r in prop_info['ranges']])
                    f.write(f"    rdfs:range [ rdf:type owl:Class ; owl:unionOf ( {ranges} ) ] .\n\n")
                else:
                    f.write("    rdfs:range owl:Thing .\n\n")
            
            # 子类关系
            f.write("# 子类关系\n")
            for subject, predicate, obj in subclass_relations:
                f.write(f"{self.namespace_prefix}:{subject} rdfs:subClassOf {self.namespace_prefix}:{obj} .\n")
            
            # 对象关系实例
            f.write("\n# 对象关系\n")
            for subject, predicate, obj in object_relations:
                f.write(f"{self.namespace_prefix}:{subject} {self.namespace_prefix}:{predicate} {self.namespace_prefix}:{obj} .\n")
        
        self.logger.info(f"Turtle文件生成完成: {output_file}")
    
    def _generate_owl_xml(self, classes: Dict[str, Dict], properties: Dict[str, Set], 
                         object_relations: List[Tuple[str, str, str]], subclass_relations: List[Tuple[str, str, str]], 
                         output_file: str, metadata: Dict):
        """
        生成OWL/XML格式文件
        """
        self.logger.info(f"生成OWL/XML格式: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # XML头和根元素
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<!DOCTYPE rdf:RDF [\n')
            f.write('    <!ENTITY owl "http://www.w3.org/2002/07/owl#" >\n')
            f.write('    <!ENTITY rdf "http://www.w3.org/1999/02/22-rdf-syntax-ns#" >\n')
            f.write('    <!ENTITY rdfs "http://www.w3.org/2000/01/rdf-schema#" >\n')
            f.write('    <!ENTITY xsd "http://www.w3.org/2001/XMLSchema#" >\n')
            f.write(f'    <!ENTITY ontoqa "{self.namespace_uri}" >\n')
            f.write(']>\n\n')
            
            f.write('<rdf:RDF xmlns="&ontoqa;"\n')
            f.write('     xml:base="&ontoqa;"\n')
            f.write('     xmlns:rdf="&rdf;"\n')
            f.write('     xmlns:owl="&owl;"\n')
            f.write('     xmlns:rdfs="&rdfs;"\n')
            f.write('     xmlns:ontoqa="&ontoqa;">\n\n')
            
            # 本体声明
            f.write('    <!-- 本体元数据 -->\n')
            f.write(f'    <owl:Ontology rdf:about="{self.namespace_uri[:-1]}">\n')
            f.write('        <rdfs:label>OntoQA Research Ontology</rdfs:label>\n')
            f.write('        <rdfs:comment>从文档聚类中提取的领域本体，用于OntoQA研究项目</rdfs:comment>\n')
            f.write(f'        <ontoqa:generatedFrom>{metadata.get("corpus_file", "unknown")}</ontoqa:generatedFrom>\n')
            f.write(f'        <ontoqa:documentCount rdf:datatype="&xsd;integer">{metadata.get("document_count", 0)}</ontoqa:documentCount>\n')
            f.write(f'        <ontoqa:clusterCount rdf:datatype="&xsd;integer">{metadata.get("cluster_count", 0)}</ontoqa:clusterCount>\n')
            f.write(f'        <ontoqa:ontologyCount rdf:datatype="&xsd;integer">{metadata.get("ontology_count", 0)}</ontoqa:ontologyCount>\n')
            f.write('    </owl:Ontology>\n\n')
            
            # 类定义
            f.write('    <!-- 类定义 -->\n')
            for class_name, class_info in classes.items():
                f.write(f'    <owl:Class rdf:about="#{class_name}">\n')
                f.write(f'        <rdfs:label>{self._xml_escape(class_info["original_name"])}</rdfs:label>\n')
                if class_info['description']:
                    f.write(f'        <rdfs:comment>{self._xml_escape(class_info["description"])}</rdfs:comment>\n')
                f.write(f'        <ontoqa:documentCount rdf:datatype="&xsd;integer">{class_info["document_count"]}</ontoqa:documentCount>\n')
                f.write(f'        <ontoqa:clusterId rdf:datatype="&xsd;integer">{class_info["cluster_id"]}</ontoqa:clusterId>\n')
                f.write('    </owl:Class>\n\n')
            
            # 数据属性定义
            f.write('    <!-- 数据属性定义 -->\n')
            data_properties = [
                ("documentCount", "文档数量", "&xsd;integer"),
                ("clusterId", "聚类ID", "&xsd;integer"),
                ("generatedFrom", "生成源文件", "&xsd;string"),
                ("clusterCount", "聚类数量", "&xsd;integer"),
                ("ontologyCount", "本体数量", "&xsd;integer")
            ]
            
            for prop_name, prop_label, prop_range in data_properties:
                f.write(f'    <owl:DatatypeProperty rdf:about="#{prop_name}">\n')
                f.write(f'        <rdfs:label>{prop_label}</rdfs:label>\n')
                f.write(f'        <rdfs:range rdf:resource="{prop_range}"/>\n')
                f.write('    </owl:DatatypeProperty>\n\n')
            
            # 对象属性定义
            f.write('    <!-- 对象属性定义 -->\n')
            for prop_name, prop_info in properties.items():
                f.write(f'    <owl:ObjectProperty rdf:about="#{prop_name}">\n')
                f.write(f'        <rdfs:label>{prop_name}</rdfs:label>\n')
                
                # 定义域
                if len(prop_info['domains']) == 1:
                    domain = list(prop_info['domains'])[0]
                    f.write(f'        <rdfs:domain rdf:resource="#{domain}"/>\n')
                elif len(prop_info['domains']) > 1:
                    f.write('        <rdfs:domain>\n')
                    f.write('            <owl:Class>\n')
                    f.write('                <owl:unionOf rdf:parseType="Collection">\n')
                    for domain in prop_info['domains']:
                        f.write(f'                    <owl:Class rdf:about="#{domain}"/>\n')
                    f.write('                </owl:unionOf>\n')
                    f.write('            </owl:Class>\n')
                    f.write('        </rdfs:domain>\n')
                
                # 值域
                if len(prop_info['ranges']) == 1:
                    range_class = list(prop_info['ranges'])[0]
                    f.write(f'        <rdfs:range rdf:resource="#{range_class}"/>\n')
                elif len(prop_info['ranges']) > 1:
                    f.write('        <rdfs:range>\n')
                    f.write('            <owl:Class>\n')
                    f.write('                <owl:unionOf rdf:parseType="Collection">\n')
                    for range_class in prop_info['ranges']:
                        f.write(f'                    <owl:Class rdf:about="#{range_class}"/>\n')
                    f.write('                </owl:unionOf>\n')
                    f.write('            </owl:Class>\n')
                    f.write('        </rdfs:range>\n')
                
                f.write('    </owl:ObjectProperty>\n\n')
            
            # 子类关系
            f.write('    <!-- 子类关系 -->\n')
            for subject, predicate, obj in subclass_relations:
                f.write(f'    <rdf:Description rdf:about="#{subject}">\n')
                f.write(f'        <rdfs:subClassOf rdf:resource="#{obj}"/>\n')
                f.write('    </rdf:Description>\n\n')
            
            # 对象关系
            f.write('    <!-- 对象关系 -->\n')
            for subject, predicate, obj in object_relations:
                f.write(f'    <rdf:Description rdf:about="#{subject}">\n')
                f.write(f'        <{predicate} rdf:resource="#{obj}"/>\n')
                f.write('    </rdf:Description>\n\n')
            
            f.write('</rdf:RDF>\n')
        
        self.logger.info(f"OWL/XML文件生成完成: {output_file}")
    
    def _xml_escape(self, text: str) -> str:
        """
        XML特殊字符转义
        """
        return (text.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;')
                   .replace("'", '&apos;'))


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='将JSON本体转换为OWL格式')
    parser.add_argument('json_file', help='输入的JSON本体文件路径')
    parser.add_argument('--output-dir', default='results/owl', help='输出目录（默认: results/owl）')
    parser.add_argument('--namespace', default='http://ontoqa.org/ontology#', 
                       help='本体命名空间URI（自动添加#结尾）')
    parser.add_argument('--format', choices=['turtle', 'xml', 'all'], nargs='+', 
                       default=['all'], help='输出格式选择（默认: all）')
    parser.add_argument('--verbose', action='store_true', help='详细日志输出')
    
    args = parser.parse_args()
    
    # 自动修正命名空间URI
    namespace_uri = args.namespace
    if not namespace_uri.endswith('#') and not namespace_uri.endswith('/'):
        namespace_uri += '#'
        print(f"自动修正命名空间为: {namespace_uri}")
    
    # 配置日志（仅在主程序运行时）
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    try:
        # 创建转换器并执行转换
        converter = JSONToOWLConverter(namespace_uri=namespace_uri)
        result = converter.convert_json_to_owl(args.json_file, args.output_dir, args.format)
        
        print(f"转换完成！")
        for format_type, file_path in result.items():
            format_name = "Turtle格式" if format_type == "turtle" else "OWL/XML格式"
            print(f"{format_name}: {file_path}")
            
    except Exception as e:
        print(f"转换失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    main()