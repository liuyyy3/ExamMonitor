# encoding: utf-8
# @File  : detection_loop.py
# @Author: Xinghui
# @Date  : 2025/12/5 15:19

# 后台检测线程主循环（读视频 + 跑模型）,检测线程，内部调用 push_event()

# app/detection_loop.py
import threading
import time
import queue
import cv2
from Cython.Build.Dependencies import normalize_existing

from app.config import Config
from models.pose_infer import PoseInfer
from models.headturn_cls import HeadTurnClassifier
from behaviors.head_turn import detect_head_turns
from behaviors.raise_hand import detect_raise_hands

from utils.json_schema import make_abnormal_frame
from socket_sever.event_sever import broadcast_event

# 事件队列：routes.py 从这里取事件
_event_queue: queue.Queue = queue.Queue()

# 当前状态：给 /status 用
_current_abnormal = False
_current_boxes = []  # 当前异常框（list[dict]），结构和 event 里的 boxes 相同

# 保护状态的锁
_state_lock = threading.Lock()


def push_event(event: dict):
    # 内部用：压入事件队列, HTTP用法
    _event_queue.put(event)

def pop_all_events():
    # routes 调用：把当前所有事件全部取出并清空， 给/api/events 使用
    events = []
    while True:
        try:
            events.append(_event_queue.get_nowait())
        except queue.Empty:
            break
    return events

def _update_state(abnormal: bool, boxes):
    global _current_abnormal, _current_boxes
    with _state_lock:
        _current_abnormal = abnormal
        _current_boxes = boxes if abnormal else []

def get_current_state():
    """routes 调用：返回当前状态（只读）"""
    with _state_lock:
        return {
            "abnormal": _current_abnormal,
            "boxes": list(_current_boxes),  # 拷贝一份，避免被外面改
        }

def _detection_loop(cfg: Config):
    # 后台检测主循环：打开 RTSP, 调用 YOLO-pose + MobileNet 判定转头, 只在“从无异常 -> 有异常”和“从有异常 -> 无异常”时发事件 JSON

    # 初始化模型
    pose_model = PoseInfer(
        cfg.POSE_RKNN_PATH,
        input_size=cfg.POSE_INPUT_SIZE,
        bg_color=cfg.POSE_BG_COLOR,
        pose_conf_thr=cfg.POSE_CONF_THR,
        nms_thr=cfg.NMS_THRESH,
    )

    cls_model = HeadTurnClassifier(
        cfg.CLS_RKNN_PATH,
        class_names=cfg.CLASS_NAMES,
    )

    frame_idx = 0
    last_msg_time = 0.0

    while True:
        cap = cv2.VideoCapture(cfg.RTSP_URL)
        if not cap.isOpened():
            print(f"[DetectionLoop] RTSP 打不开，{cfg.RECONNECT_INTERVAL_SEC}s 后重试: {cfg.RTSP_URL}")
            time.sleep(cfg.RECONNECT_INTERVAL_SEC)
            continue

        print("[DetectionLoop] RTSP 打开成功，开始检测")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[DetectionLoop] 读取帧失败，准备重连 RTSP ...")
                cap.release()
                time.sleep(cfg.RECONNECT_INTERVAL_SEC)
                break  # 跳出内层 while，重新走外层，重连 RTSP

            frame_idx += 1
            # 跳帧检测
            if cfg.FRAME_STRIDE > 1 and (frame_idx % cfg.FRAME_STRIDE != 0):
                continue

            # 1) YOLO-pose 检测（只返回 person + 关键点）
            pose_results = pose_model.infer(frame)

            # 用骨架 + 分类器 做转头判断
            head_boxes, _ = detect_head_turns(
                frame,
                pose_results,
                cls_model,
                kp_conf_thr=cfg.KP_CONF_THR,
                min_valid_kpts=cfg.MIN_VALID_KPTS,
                head_turn_prob_thr=cfg.HEAD_TURN_PROB_THR,
            )

            # 根据关键点位置判断举手
            raise_boxes, total_person_rh = detect_raise_hands(
                frame,
                pose_results,
                kp_conf_thr=cfg.KP_CONF_THR,
            )

            all_boxes = head_boxes + raise_boxes
            # 更新当前状态，给 /status 用
            _update_state(
                abnormal = (len(all_boxes) > 0),
                boxes = all_boxes,
            )

            # 按时间节流发 json，不管有无异常都发送
            now_ts = time.time()
            if now_ts - last_msg_time >= cfg.FRAME_MSG_INTERVAL:
                msg = make_abnormal_frame(all_boxes)
                push_event(msg)  # 支持 HTTP调试使用
                broadcast_event(msg)
                last_msg_time = now_ts


def start_detection_thread(cfg: Config):
    # 在 run_server.py 里调用，起一个守护线程跑 _detection_loop
    t = threading.Thread(target=_detection_loop, args=(cfg,), daemon=True)
    t.start()
    print("[Main] Detection thread started.")

