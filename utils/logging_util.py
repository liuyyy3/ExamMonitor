# encoding: utf-8
# @File  : logging_util.py
# @Author: Xinghui
# @Date  : 2025/12/31 11:27

import logging
import re
from pathlib import Path

_LOGGER_NAME = "ExamMonitor"

def _next_log_path(log_dir: Path, prefix: str = "ExamPyLog", suffix: str = ".txt") -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(suffix)}$")
    max_num = 0
    for entry in log_dir.iterdir():
        if not entry.is_file():
            continue
        match = pattern.match(entry.name)
        if not match:
            continue
        try:
            num = int(match.group(1))
        except ValueError:
            continue
        max_num = max(max_num, num)
    next_num = max_num + 1
    return log_dir / f"{prefix}{next_num}{suffix}"

def setup_run_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger

    log_dir = Path("/home/cat/ExamMonitor/PyLogs")
    log_path = _next_log_path(log_dir)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    formatter = logging.Formatter("%(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info(f"[Log] Start logging to : {log_path}")

    return logger

def get_logger() -> logging.Logger:
    return logging.getLogger(_LOGGER_NAME)





