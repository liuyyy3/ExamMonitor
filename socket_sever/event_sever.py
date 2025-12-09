# encoding: utf-8
# @File  : event_sever.py
# @Author: Xinghui
# @Date  : 2025/12/8 9:47

# 新增 TCP/UDP 服务器，负责把事件发送给前端

import socket
import threading
import json

_client_sockets = []
_lock = threading.Lock()

def start_event_sever(host = "0.0.0.0", port = 6000):
    t = threading.Thread(target=_sever_loop, args = (host, port), daemon = True)
    t.start()
    print(f"[SocketSever] TCP event sever started at {host}:{port}")

def _sever_loop(host, port):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(5)
    print(f"[SocketSever] TCP socket listening at {host}:{port}")

    while True:
        conn, addr = srv.accept()
        print(f"[SocketSever] Connected with {addr}")
        conn.setblocking(True)
        with _lock:
            _client_sockets.append(conn)

def broadcast_event(event: dict):
    data = (json.dumps(event, ensure_ascii = False) + "\n").encode("utf-8")
    dead = []

    with _lock:
        for s in _client_sockets:
            try:
                s.sendall(data)
            except Exception as e:
                dead.append(s)

        for s in dead:
            try:
                s.close()
            except Exception:
                pass
            _client_sockets.remove(s)


