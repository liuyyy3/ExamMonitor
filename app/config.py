# encoding: utf-8
# @File  : config.py
# @Author: Xinghui
# @Date  : 2025/12/5 15:15

# 配置（视频源、模型路径、阈值等）


class Config:
    RTSP_URL = ""
    POSE_RKNN_PATH = '/home/tom/test_program/model/yolov8n-pose.rknn'
    CLS_RKNN_PATH = '/home/tom/test_program/model/MobileNetV2_1.rknn'
    CLASS_NAMES = ['normal', 'turn_head']

    # 5) YOLO-pose 阈值
    POSE_INPUT_SIZE = (640, 640)
    POSE_BG_COLOR = 56
    POSE_CONF_THR = 0.3  # 对应 objectThresh
    NMS_THRESH = 0.4

    # 7) 关键点质量筛选
    KP_CONF_THR = 0.2
    MIN_VALID_KPTS = 3  # 上半身 0~12 中，大于阈值的点数

    HEAD_TURN_PROB_THR = 0.5

    # 8) 跳帧（例如 2 表示处理 1 帧跳 1 帧；=1 就是每帧都处理）
    FRAME_STRIDE = 1

    # RTSP 断流时重连等待秒数
    RECONNECT_INTERVAL_SEC = 5

config = Config()

