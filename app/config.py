# encoding: utf-8
# @File  : config.py
# @Author: Xinghui
# @Date  : 2025/12/5 15:15

# 配置（视频源、模型路径、阈值等）
import os
from pathlib import Path

class Config:

    # Node服务地址
    NODE_BASE = os.getenv('NODE_BASE', "http://192.168.9.50:8080")  # NODE_BASE要换成node的实际地址
    # 具体接口以后定，先用 warning
    REPORT_URL = f"{NODE_BASE}/warning/alg_alarm_fetch"
    # 鉴定权限，如果 node需要
    TOKEN = os.getenv("NODE_TOKEN", "")  # 前端的告警上报接口，从这里上报告警信息，node需要校验 token

    # 截图保存目录
    SNAP_DIR = Path(os.getenv("EXAM_SNAP_DIR", "/home/tom/ExamMonitor/snapshots"))
    CAMERA_NAME = os.getenv("EXAM_CAMERA_NAME", "exam_room1")

    # 基础配置信息
    RTSP_URL = "rtsp://192.168.9.140:8554/stream1"
    POSE_RKNN_PATH = '/home/tom/test_program/model/yolov8n-pose.rknn'
    CLS_RKNN_PATH = '/home/tom/test_program/model/mobilenet_cs12_fp32_nonorm.rknn'
    CLASS_NAMES = ['normal', 'turn_head']

    # 上传json文件的间隔，为了保持前端画的框跟得上人物
    FRAME_MSG_INTERVAL = 1.0

    # YOLO-pose 阈值
    POSE_INPUT_SIZE = (640, 640)
    POSE_BG_COLOR = 56
    POSE_CONF_THR = 0.3  # 对应 objectThresh
    NMS_THRESH = 0.4

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

config = Config()

