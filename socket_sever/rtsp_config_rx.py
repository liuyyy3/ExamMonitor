# encoding: utf-8
# @File  : rtsp_config_rx.py
# @Author: Xinghui
# @Date  : 2025/12/15 17:29

# 加入组播 并且 收包解析 JSON

"""
import socket
import json
import struct
import threading
import re
from typing import Optional, Any

from .rtsp_state import set_rtsp_url

MCAST_GROUP = "127.0.0.1"
MCAST_PORT = 6001


# 从任意字符串里抓 RTSP（尽量宽松，但不包含空白/引号/右括号等终止符）
_RTSP_RE = re.compile(r"(rtsps?://[^\s\"\'\)\]\}\>,]+)", re.IGNORECASE)

def _make_mcast_rx_socket(bind_ip: str = "0.0.0.0") -> socket.socket:
    # Linux / RK3588：绑定端口并加入组播。
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        # 有些系统支持 REUSEPORT
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    except OSError:
        pass
    sock.bind((bind_ip, MCAST_PORT))

    # 加入组播组，一起愉快的玩耍
    mreq = struct.pack("4s4s", socket.inet_aton(MCAST_GROUP), socket.inet_aton("0.0.0.0"))
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    return sock

def _extract_rtsp_from_any(obj: Any) -> Optional[str]:
    # 从任意 Python 对象（dict/list/str/...）中提取第一个 rtsp://... 字符串
    if obj is None:
        return None

    # 字符串：直接 regex
    if isinstance(obj, str):
        m = _RTSP_RE.search(obj)
        if not m:
            return None
        rtsp_url = m.group()
        if isinstance(rtsp_url, tuple):
            rtsp_url = rtsp_url[0]
        return str(rtsp_url).strip()

    # dict：先优先看常见 key，再递归遍历 values
    if isinstance(obj, dict):
        # 兼容常见字段名
        for k in ("rtsp_url", "rtsp", "url", "mediaUrl", "MediaUrl", "stream", "playUrl", "RTSP"):
            v = obj.get(k)
            r = _extract_rtsp_from_any(v)
            if r:
                return r
        # 不管字段叫什么，遍历所有 value
        for v in obj.values():
            r = _extract_rtsp_from_any(v)
            if r:
                return r
        return None

    # list/tuple：逐个递归
    if isinstance(obj, (list, tuple)):
        for it in obj:
            r = _extract_rtsp_from_any(it)
            if r:
                return r
        return None

    # 其它类型：转字符串再试一次
    try:
        s = str(obj)
    except Exception:
        return None
    return _extract_rtsp_from_any(s)


def start_rtsp_config_rx(stop_event: threading.Event, bind_ip: str = "0.0.0.0") -> threading.Thread:
    # 后台线程：接收前端发来的 rtsp_config，解析出 rtsp_url 并保存
    def _worker():
        sock: Optional[socket.socket] = None
        try:
            sock = _make_mcast_rx_socket(bind_ip)
            sock.settimeout(1)
            print(f"[RTSP_RX] listening {MCAST_GROUP}:{MCAST_PORT} ...")

            while not stop_event.is_set():
                try:
                    data, addr = sock.recvfrom(64 * 1024)
                    print(f"[RTSP_RX] 收到数据，长度: {len(data)}，来自: {addr}")

                except socket.timeout:
                    continue
                except OSError as e:
                    print(f"[RTSP_RX] socket错误: {e}")
                    break

                txt = data.decode("utf-8", errors="ignore").strip()
                # 先尝试 JSON
                msg_obj = None
                try:
                    msg_obj = json.loads(txt)
                except Exception:
                    msg_obj = None

                rtsp_url = None
                if msg_obj is not None:
                    rtsp_url = _extract_rtsp_from_any(msg_obj)
                if not rtsp_url:
                    rtsp_url = _extract_rtsp_from_any(txt)

                if rtsp_url:
                    set_rtsp_url(rtsp_url)
                    print(f"[RTSP_RX] from {addr} => {rtsp_url}")
                else:
                    # debug：没找到就打印前 120 字符，方便你看前端到底发了啥
                    print(f"[RTSP_RX] 未解析到RTSP，raw={txt[:120]}")

        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass

    t = threading.Thread(target=_worker, name="start_rtsp_config_rx", daemon=True)
    t.start()
    return t
"""


# socket_sever/rtsp_config_rx.py
import socket
import json
import threading
import re
from typing import Optional, Any

from .rtsp_state import set_rtsp_url

BIND_IP = "127.0.0.1"
PORT = 6001

_RTSP_RE = re.compile(r"(rtsps?://[^\s\"\'\)\]\}\>,]+)", re.IGNORECASE)


def _extract_rtsp_from_any(obj: Any) -> Optional[str]:
    if obj is None:
        return None

    if isinstance(obj, str):
        m = _RTSP_RE.search(obj)
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

