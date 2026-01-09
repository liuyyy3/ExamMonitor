# encoding: utf-8
# @File  : rtsp_config_rx.py
# @Author: Xinghui
# @Date  : 2025/12/15 17:29

# 加入组播 并且 收包解析 JSON

import socket
import json
import threading
import re
from typing import Optional, Any

from .rtsp_state import set_rtsp_url

BIND_IP = "127.0.0.1"
PORT = 6001

# _RTSP_RE = re.compile(r"(rtsps?://[^\s\"\'\)\]\}\>,]+)", re.IGNORECASE)
_STREAM_RE = re.compile(r"((?:rtsp?|udp)://[^\s\"\'\)\]\}\>,]+)", re.IGNORECASE)

def _extract_rtsp_from_any(obj: Any) -> Optional[str]:
    if obj is None:
        return None

    if isinstance(obj, str):
        m = _STREAM_RE.search(obj)
        if not m:
            return None
        return m.group(0).strip()

    if isinstance(obj, dict):
        for k in ("rtsp_url", "rtsp", "url", "mediaUrl", "MediaUrl", "stream", "playUrl", "RTSP"):
            r = _extract_rtsp_from_any(obj.get(k))
            if r:
                return r
        for v in obj.values():
            r = _extract_rtsp_from_any(v)
            if r:
                return r
        return None

    if isinstance(obj, (list, tuple)):
        for it in obj:
            r = _extract_rtsp_from_any(it)
            if r:
                return r
        return None

    return _extract_rtsp_from_any(str(obj))


def start_rtsp_config_rx(stop_event: threading.Event) -> threading.Thread:
    def _worker():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((BIND_IP, PORT))
        print(f"[RTSP_RX] listening UDP {BIND_IP}:{PORT}")

        while not stop_event.is_set():
            try:
                data, addr = sock.recvfrom(65535)
            except Exception as e:
                print(f"[RTSP_RX] recv error: {e}")
                continue

            txt = data.decode("utf-8", errors="ignore").strip()
            print(f"[RTSP_RX] raw from {addr}: {txt[:120]}")

            # 先尝试按 JSON 解析
            obj = None
            try:
                obj = json.loads(txt)
            except Exception:
                obj = None

            rtsp_url = _extract_rtsp_from_any(obj) if obj is not None else None
            if not rtsp_url:
                rtsp_url = _extract_rtsp_from_any(txt)

            if rtsp_url:
                set_rtsp_url(rtsp_url)
                print(f"[RTSP_RX] set rtsp_url = {rtsp_url}")
            else:
                print("[RTSP_RX] no rtsp found")

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t
