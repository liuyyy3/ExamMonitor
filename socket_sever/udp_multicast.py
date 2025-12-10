# encoding: utf-8
# @File  : udp_multicast.py
# @Author: Xinghui
# @Date  : 2025/12/9 18:07

import socket
import json

# 默认组播地址和端口
MULTICAST_GROUP = "239.0.0.10"   # 组播地址
MULTICAST_PORT = 6000           # 端口号

_sock = None


def init_multicast(group: str = None, port: int = None, ttl: int = 1):
    # 在 run_server.py 里调用一次，用来初始化 UDP 组播 socket。
    # group: 组播 IP， port: 端口， ttl: 组播包的 TTL，1 表示只在本地网段，不出网关
    # global _sock, MULTICAST_GROUP, MULTICAST_PORT

    if group is not None:
        MULTICAST_GROUP = group
    if port is not None:
        MULTICAST_PORT = port

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

    # 设置 TTL
    sock.setsockopt(
        socket.IPPROTO_IP,
        socket.IP_MULTICAST_TTL,
        ttl.to_bytes(1, byteorder="big"),
    )

    # 设置本地网卡：
    # sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton("本机IP"))

    _sock = sock
    print(f"[UDP-MCAST] init ok, group={MULTICAST_GROUP}, port={MULTICAST_PORT}, ttl={ttl}")


def send_json(msg: dict):
    # 把字典转成 JSON + 换行，通过 UDP 组播发出去

    global _sock
    if _sock is None:
        raise RuntimeError("UDP multicast socket not initialized, call init_multicast() first")

    data = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
    _sock.sendto(data, (MULTICAST_GROUP, MULTICAST_PORT))
