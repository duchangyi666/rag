"""
FinRAG 日志模块

从 base.config 读取 [log] section 参数，配置控制台 + RotatingFileHandler 双通道日志。

Usage:
    from base.logger import setup_logger, single_logger

    single_logger.info("RAG pipeline started")

    # 子模块 logger
    logger = setup_logger("fin_rag.mysql_qa")
    logger.error("DB connection failed", exc_info=True)
"""

import logging
import os
from logging.handlers import RotatingFileHandler

from base.config import config


def setup_logger(name: str = None, log_file: str = None) -> logging.Logger:
    """
    创建并配置日志记录器。

    Args:
        name: Logger 名称，默认使用 config.log_name（即 config.ini [log] name 的值）。
        log_file: 日志文件完整路径，默认使用 config.log_filepath。
    """
    # ── 从 config 读取参数 ────────────────────────────────────────
    if name is None:
        name = config.log_name

    if log_file is None:
        log_file = config.log_filepath

    # ── 确保日志目录存在 ──────────────────────────────────────────
    log_dir = os.path.dirname(os.path.abspath(log_file))
    os.makedirs(log_dir, exist_ok=True)

    # ── 创建 logger ───────────────────────────────────────────────
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # ── 避免重复添加 handler ─────────────────────────────────────
    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    has_file = any(
        isinstance(h, RotatingFileHandler)
        for h in logger.handlers
    )

    console_level = _parse_level(config.log_console_level, logging.INFO)
    file_level = _parse_level(config.log_file_level, logging.DEBUG)
    formatter = logging.Formatter(fmt=config.log_format, datefmt=config.log_date_format)

    # ── 控制台处理器 ──────────────────────────────────────────────
    if config.log_enable_console and not has_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # ── 文件处理器（按大小轮转）──────────────────────────────────
    if not has_file:
        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=config.log_max_bytes_mb * 1024 * 1024,
            backupCount=config.log_backup_count,
            encoding=config.log_encoding,
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # ── 有独立 handler 时关闭向上传播，避免重复输出 ──────────────
    if logger.handlers:
        logger.propagate = False

    return logger


def _parse_level(level_str: str, default: int) -> int:
    """将 "INFO" / "DEBUG" 等字符串转为 logging 常量。"""
    try:
        return getattr(logging, level_str.strip().upper())
    except AttributeError:
        return default


# ── 模块级默认 logger ──────────────────────────────────────────────
single_logger = setup_logger()
