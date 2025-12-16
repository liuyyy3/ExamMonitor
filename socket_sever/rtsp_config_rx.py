# encoding: utf-8
# @File  : rtsp_config_rx.py
# @Author: Xinghui
# @Date  : 2025/12/15 17:29

# 加入组播 并且 收包解析 JSON

import socket
import json
import struct
import threading
from typing import Optional

from .rtsp_state import set_rtsp_url

MCAST_GROUP = "239.0.0.10"
MCAST_PORT = 6001

def _make_mcast_rx_socket(bind_ip: str = "0.0.0.0") -> socket.socket:
    # Linux / RK3588：绑定端口并加入组播。
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    except OSError:
        pass
    sock.bind((bind_ip, MCAST_PORT))

    # 加入组播组，一起愉快的玩耍
    mreq = struct.pack("4s4s", socket.inet_aton(MCAST_GROUP), socket.inet_aton("0.0.0.0"))
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    return sock

def start_rtsp_config_rx(stop_event: threading.Event, bind_ip: str = "0.0.0.0") -> threading.Thread:
    # 后台线程：接收前端发来的 rtsp_config，提取 rtsp_url 并保存
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

                try:
                    txt = data.decode("utf-8", errors="ignore").strip()
                    msg = json.loads(txt)
                except Exception:
                    continue

                if not isinstance(msg, dict):
                    continue
                if msg.get("msg_type") != "rtsp_config":
                    continue

                rtsp_url = msg.get("rtsp_url")
                if isinstance(rtsp_url, str) and rtsp_url.strip():
                    set_rtsp_url(rtsp_url)
                    print(f"[RTSP_RX] from {addr} => {rtsp_url}")

        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass

    t = threading.Thread(target=_worker, name="start_rtsp_config_rx", daemon=True)
    t.start()
    return t

