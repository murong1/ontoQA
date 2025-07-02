#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结果文件清理脚本
清理过期的结果文件，保留最新的N个运行结果
"""

import argparse
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple


def setup_logging(verbose: bool = False):
    """设置日志配置"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def get_timestamped_files(directory: Path, pattern: str) -> List[Tuple[datetime, Path]]:
    """
    获取带时间戳的文件列表
    
    Args:
        directory: 目录路径
        pattern: 文件模式，包含时间戳占位符，如 "*_YYYYMMDD_HHMMSS.*"
    
    Returns:
        List of (timestamp, filepath) tuples, sorted by timestamp (newest first)
    """
    files_with_timestamp = []
    
    for file_path in directory.glob(pattern):
        try:
            # 从文件名中提取时间戳
            filename = file_path.name
            # 假设时间戳格式为 YYYYMMDD_HHMMSS
            import re
            timestamp_match = re.search(r'(\d{8}_\d{6})', filename)
            if timestamp_match:
                timestamp_str = timestamp_match.group(1)
                timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                files_with_timestamp.append((timestamp, file_path))
        except ValueError:
            # 无法解析时间戳，跳过
            continue
    
    # 按时间戳排序，最新的在前
    files_with_timestamp.sort(key=lambda x: x[0], reverse=True)
    return files_with_timestamp


def get_run_directories(results_dir: Path) -> List[Tuple[datetime, Path]]:
    """
    获取运行目录列表
    
    Returns:
        List of (timestamp, dirpath) tuples, sorted by timestamp (newest first)
    """
    run_dirs = []
    
    for dir_path in results_dir.glob("run_*"):
        if dir_path.is_dir():
            try:
                # 从目录名中提取时间戳
                dir_name = dir_path.name
                if dir_name.startswith("run_"):
                    timestamp_str = dir_name[4:]  # 移除 "run_" 前缀
                    timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    run_dirs.append((timestamp, dir_path))
            except ValueError:
                # 无法解析时间戳，跳过
                continue
    
    # 按时间戳排序，最新的在前
    run_dirs.sort(key=lambda x: x[0], reverse=True)
    return run_dirs


def cleanup_old_files(results_dir: Path, keep_count: int = 5, dry_run: bool = False) -> dict:
    """
    清理旧的结果文件
    
    Args:
        results_dir: results目录路径
        keep_count: 保留的文件/目录数量
        dry_run: 仅显示要删除的文件，不实际删除
    
    Returns:
        清理统计信息
    """
    logger = logging.getLogger(__name__)
    stats = {
        'run_dirs_deleted': 0,
        'old_files_deleted': 0,
        'space_freed': 0
    }
    
    if not results_dir.exists():
        logger.warning(f"结果目录不存在: {results_dir}")
        return stats
    
    # 1. 清理运行目录
    logger.info("清理旧的运行目录...")
    run_dirs = get_run_directories(results_dir)
    
    if len(run_dirs) > keep_count:
        dirs_to_delete = run_dirs[keep_count:]
        logger.info(f"找到 {len(run_dirs)} 个运行目录，将删除最旧的 {len(dirs_to_delete)} 个")
        
        for timestamp, dir_path in dirs_to_delete:
            if dry_run:
                logger.info(f"[DRY RUN] 将删除目录: {dir_path} ({timestamp})")
            else:
                try:
                    # 计算目录大小
                    dir_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                    
                    shutil.rmtree(dir_path)
                    logger.info(f"已删除目录: {dir_path} ({timestamp})")
                    stats['run_dirs_deleted'] += 1
                    stats['space_freed'] += dir_size
                except Exception as e:
                    logger.error(f"删除目录失败 {dir_path}: {e}")
    else:
        logger.info(f"运行目录数量 ({len(run_dirs)}) 未超过保留限制 ({keep_count})，无需清理")
    
    # 2. 清理根目录下的旧文件
    logger.info("清理根目录下的旧文件...")
    
    # 定义要清理的文件模式
    file_patterns = [
        "clustering_details_*.json",
        "evaluation_results_*.json", 
        "evaluation_summary_*.txt",
        "ontology_generation_details_*.json",
        "ontology_processing_details_*.json",
        "retrieval_details_*.json"
    ]
    
    # 注意：新的文件结构将详情文件放在run_*/debug/目录下，
    # 这些文件会随着运行目录一起被清理，无需单独处理
    
    for pattern in file_patterns:
        files_with_timestamp = get_timestamped_files(results_dir, pattern)
        
        if len(files_with_timestamp) > keep_count:
            files_to_delete = files_with_timestamp[keep_count:]
            logger.info(f"模式 '{pattern}': 找到 {len(files_with_timestamp)} 个文件，将删除最旧的 {len(files_to_delete)} 个")
            
            for timestamp, file_path in files_to_delete:
                if dry_run:
                    logger.info(f"[DRY RUN] 将删除文件: {file_path} ({timestamp})")
                else:
                    try:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        logger.debug(f"已删除文件: {file_path} ({timestamp})")
                        stats['old_files_deleted'] += 1
                        stats['space_freed'] += file_size
                    except Exception as e:
                        logger.error(f"删除文件失败 {file_path}: {e}")
        else:
            logger.debug(f"模式 '{pattern}': 文件数量 ({len(files_with_timestamp)}) 未超过保留限制 ({keep_count})")
    
    return stats


def cleanup_by_age(results_dir: Path, days: int = 30, dry_run: bool = False) -> dict:
    """
    按时间清理文件（删除超过指定天数的文件）
    
    Args:
        results_dir: results目录路径
        days: 保留天数
        dry_run: 仅显示要删除的文件，不实际删除
    
    Returns:
        清理统计信息
    """
    logger = logging.getLogger(__name__)
    stats = {'files_deleted': 0, 'dirs_deleted': 0, 'space_freed': 0}
    
    cutoff_date = datetime.now() - timedelta(days=days)
    logger.info(f"删除 {cutoff_date.strftime('%Y-%m-%d')} 之前的文件")
    
    # 清理运行目录
    run_dirs = get_run_directories(results_dir)
    for timestamp, dir_path in run_dirs:
        if timestamp < cutoff_date:
            if dry_run:
                logger.info(f"[DRY RUN] 将删除目录: {dir_path} ({timestamp})")
            else:
                try:
                    dir_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                    shutil.rmtree(dir_path)
                    logger.info(f"已删除目录: {dir_path} ({timestamp})")
                    stats['dirs_deleted'] += 1
                    stats['space_freed'] += dir_size
                except Exception as e:
                    logger.error(f"删除目录失败 {dir_path}: {e}")
    
    return stats


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='清理OntoQA结果文件')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='结果目录路径 (默认: results)')
    parser.add_argument('--keep-count', type=int, default=5,
                       help='保留的文件/目录数量 (默认: 5)')
    parser.add_argument('--keep-days', type=int,
                       help='按天数保留文件，超过指定天数的文件将被删除')
    parser.add_argument('--dry-run', action='store_true',
                       help='仅显示要删除的文件，不实际删除')
    parser.add_argument('--verbose', action='store_true',
                       help='详细输出')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    results_dir = Path(args.results_dir)
    
    if args.dry_run:
        logger.info("=== DRY RUN 模式：仅显示要删除的文件，不会实际删除 ===")
    
    try:
        if args.keep_days:
            # 按天数清理
            stats = cleanup_by_age(results_dir, args.keep_days, args.dry_run)
            logger.info(f"按时间清理完成: 删除了 {stats['files_deleted']} 个文件, {stats['dirs_deleted']} 个目录")
        else:
            # 按数量保留
            stats = cleanup_old_files(results_dir, args.keep_count, args.dry_run)
            logger.info(f"按数量清理完成: 删除了 {stats['old_files_deleted']} 个文件, {stats['run_dirs_deleted']} 个目录")
        
        if not args.dry_run and stats.get('space_freed', 0) > 0:
            space_mb = stats['space_freed'] / (1024 * 1024)
            logger.info(f"释放空间: {space_mb:.2f} MB")
            
    except Exception as e:
        logger.error(f"清理失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())