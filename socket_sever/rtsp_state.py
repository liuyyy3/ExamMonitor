# encoding: utf-8
# @File  : rtsp_state.py
# @Author: Xinghui
# @Date  : 2025/12/15 17:11

# 监听指定 udp 端口，接收前端输入的 rtsp 路径，以做后续的检测进程

import threading
from typing import Optional

_lock = threading.Lock()
_latest_rtsp_url: Optional[str] = None

def set_rtsp_url(url: str) -> None:
    global _latest_rtsp_url
    url = (url or "").strip()
    if not url:
        return
    with _lock:
        _latest_rtsp_url = url

def get_rtsp_url() -> Optional[str]:
    with _lock:
        return _latest_rtsp_url
