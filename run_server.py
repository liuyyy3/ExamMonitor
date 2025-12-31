# encoding: utf-8
# @File  : run_server.py
# @Author: Xinghui
# @Date  : 2025/12/5 15:25

# 项目入口：启动检测线程 + Flask

from app import create_app
from app.config import config
from app.detection_loop import start_detection_thread
from socket_sever.udp_multicast import init_multicast

import threading
from socket_sever.rtsp_config_rx import start_rtsp_config_rx
from utils.logging_util import setup_run_logger

import signal
import sys
import logging

def handle_exit(sig, frame):
    logger = logging.getLogger("ExamMonitor")
    logger.info("[System] Caught signal, shutting down...")
    logging.shutdown()   # 🔥 关键：flush + close 所有 handler
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

app = create_app()

if __name__ == '__main__':
    setup_run_logger()
    init_multicast(group="127.0.0.1", port=6000, ttl=1)
    stop_event = threading.Event()
    start_rtsp_config_rx(stop_event)  # 监听 239.0.0.10:6001
    start_detection_thread(config)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
