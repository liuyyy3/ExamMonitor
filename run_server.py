# encoding: utf-8
# @File  : run_server.py
# @Author: Xinghui
# @Date  : 2025/12/5 15:25

# 项目入口：启动检测线程 + Flask

from app import create_app
from app.config import config
from app.detection_loop import start_detection_thread
from socket_sever.udp_multicast import init_multicast

app = create_app()

if __name__ == '__main__':
    init_multicast(group="239.0.0.10", port=6000, ttl=1)
    start_detection_thread(config)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
