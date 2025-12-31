# encoding: utf-8
# @File  : config.py
# @Author: Xinghui
# @Date  : 2025/12/5 15:15

# 配置（视频源、模型路径、阈值等）
import os
from pathlib import Path
import socket

def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip

class Config:

    # Node服务地址
    NODE_BASE = os.getenv('NODE_BASE', f"http://127.0.0.1:8080")  # NODE_BASE要换成node的实际地址
    # 具体接口以后定，先用 warning
    REPORT_URL = f"{NODE_BASE}/warning/alg_alarm_fetch"
    # 鉴定权限，如果 node需要
    TOKEN = os.getenv("NODE_TOKEN", "")  # 前端的告警上报接口，从这里上报告警信息，node需要校验 token

    # 截图保存目录
    SNAP_DIR = Path(os.getenv("EXAM_SNAP_DIR", "/home/cat/ExamMonitor/snapshots"))
    CAMERA_NAME = os.getenv("EXAM_CAMERA_NAME", "exam_room1")

    # 基础配置信息
    RTSP_URL = "rtsp://192.168.9.140:8554/stream1"
    POSE_RKNN_PATH = '/home/cat/ExamMonitor/models/yolov8n-pose.rknn'
    CLS_RKNN_PATH = '/home/cat/ExamMonitor/models/mobilenet_cs12_fp32_nonorm.rknn'
    CLASS_NAMES = ['normal', 'turn_head']

    # 上传json文件的间隔，为了保持前端画的框跟得上人物
    FRAME_MSG_INTERVAL = 0.15

    # YOLO-pose 阈值
    POSE_INPUT_SIZE = (640, 640)
    POSE_BG_COLOR = 56
    POSE_CONF_THR = 0.3  # 对应 objectThresh
    NMS_THRESH = 0.4

    # 告警截图标注设置参数
    LABEL_FONT_SIZE = int(os.getenv("EXAM_LABEL_FONT_SIZE", 18))   # 字体大小
    BOX_THICKNESS = int(os.getenv("EXAM_BOX_THICKNESS", 2))   # 边框粗细

    # 关键点质量筛选
    KP_CONF_THR = 0.2
    MIN_VALID_KPTS = 3  # 上半身 0~12 中，大于阈值的点数

    HEAD_TURN_PROB_THR = 0.5

    # 跳帧（例如 2 表示处理 1 帧跳 1 帧；=1 就是每帧都处理）
    FRAME_STRIDE = 1

    # RTSP 断流时重连等待秒数
    RECONNECT_INTERVAL_SEC = 5

    # 每隔多少毫秒做一次推理
    INFER_INTERVAL_MS = 100

    # 运动速度阈值（像素/秒），连续移动判定阈值
    MOVE_SPEED_PX_PER_S = float(os.getenv("EXAM_MOVE_SPEED_PX_PER_S", 160))
    MOVE_ON_N = int(os.getenv("EXAM_MOVE_ON_N", 3))  # 100ms 推理一次，连续 3 次移动才算“在走”
    MOVE_OFF_N = int(os.getenv("EXAM_MOVE_OFF_N", 6))  # 连续 6 次不动才算“停止走动”

config = Config()

