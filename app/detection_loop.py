# encoding: utf-8
# @File  : detection_loop.py
# @Author: Xinghui
# @Date  : 2025/12/5 15:19

# 后台检测线程主循环（读视频 + 跑模型）,检测线程，内部调用 push_event()

# app/detection_loop.py
import threading
import time
import queue
from pathlib import Path
import cv2

from app.config import Config
from models.pose_infer import PoseInfer
from models.headturn_cls import HeadTurnClassifier
from behaviors.head_turn import detect_head_turns
from behaviors.raise_hand import detect_raise_hands

from utils.json_schema import make_abnormal_frame
from socket_sever.udp_multicast import send_json
from utils.reporter import report_exam_alarm

# 事件队列：routes.py 从这里取事件
_event_queue: queue.Queue = queue.Queue()

# 当前状态：给 /status 用
_current_abnormal = False
_current_boxes = []  # 当前异常框（list[dict]），结构和 event 里的 boxes 相同

# 保护状态的锁
_state_lock = threading.Lock()

def _hex_to_bgr(color_str: str):
    """把 '#rrggbb' 转成 OpenCV 用的 BGR 三元组"""
    if not color_str:
        return (0, 0, 255)
    s = color_str.lstrip("#")
    if len(s) != 6:
        return (0, 0, 255)
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (b, g, r)


def _draw_boxes_on_frame(frame, boxes):
    """在图像上画出 all_boxes 里的框，颜色来自 box['color']"""
    vis = frame.copy()
    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]
        color_str = box.get("color", "#ff0000")
        color = _hex_to_bgr(color_str)
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # 如果你想在图上写文字，可以顺便把 label 也写上：
        label = box.get("type", "")
        if label:
            cv2.putText(
                vis, label,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2, cv2.LINE_AA
            )
    return vis


def _iou(b1, b2):
    x1, y1, x2, y2 = b1
    x1b, y1b, x2b, y2b = b2
    xx1 = max(x1, x1b)
    yy1 = max(y1, y1b)
    xx2 = min(x2, x2b)
    yy2 = min(y2, y2b)
    w = max(0, xx2 - xx1)
    h = max(0, yy2 - yy1)
    inter = w * h
    area1 = max(0, x2 - x1) * max(0, y2 - y1)
    area2 = max(0, x2b - x1b) * max(0, y2b - y1b)
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def _boxes_group_changed(prev_boxes, curr_boxes, iou_thr=0.5) -> bool:
    # prev_boxes / curr_boxes: list[dict]，至少含 'bbox' 和 'type'
    # 返回 True 表示“异常人群发生了变化”（需要 group_id 自增）

    if len(prev_boxes) != len(curr_boxes):
        return True

    if len(curr_boxes) == 0:
        # 都是 0 个：视作没变化（此时 group_id 保持，下一次有异常时会重新自增）
        return False

    used = [False] * len(curr_boxes)

    for p in prev_boxes:
        pb = p["bbox"]
        p_type = p.get("type")
        best_iou = 0.0
        best_idx = -1

        for j, c in enumerate(curr_boxes):
            if used[j]:
                continue
            if c.get("type") != p_type:
                continue
            iou = _iou(pb, c["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = j

        # 如果没有找到同类型且 IoU 足够大的匹配，认为换人了
        if best_idx == -1 or best_iou < iou_thr:
            return True

        used[best_idx] = True

    # 所有 prev box 都能匹配到 curr 中的某个 box → 视作“同一批人”
    return False


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

    group_id = 0  # 当前异常人群 id（自增）
    prev_boxes_for_group = []  # 上一轮用于对比分组的 boxes
    prev_count = 0  # 上一轮异常人数
    last_boxes_update_time = 0.0  # 上一次“带 boxes 的消息”时间（仅在 count>0 时更新）


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
            print(f"[DEBUG] frame_idx={frame_idx}, pose_num={len(pose_results)}")

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

            print(f"[DEBUG] head_boxes={len(head_boxes)}, raise_boxes={len(raise_boxes)}")

            all_boxes = head_boxes + raise_boxes
            curr_count = len(all_boxes)

            # 更新当前状态，给 /status 用
            _update_state(
                abnormal = (len(all_boxes) > 0),
                boxes = all_boxes,
            )

            # 按时间节流发 json，不管有无异常都发送
            now_ts = time.time()
            send_msg = False
            need_report_to_node = False

            group_change = False

            if curr_count > 0:
                if prev_count == 0:
                    group_change = True
                else:
                    if _boxes_group_changed(prev_boxes_for_group, all_boxes):
                        group_change = True

                if group_change:
                    group_id += 1
                    prev_boxes_for_group = [b.copy() for b in all_boxes]
                    need_report_to_node = True


            if curr_count != prev_count:
                send_msg = True
            elif curr_count > 0 and (now_ts - last_boxes_update_time >= cfg.FRAME_MSG_INTERVAL):
                send_msg = True

            if send_msg:
                event_json = make_abnormal_frame(all_boxes, group_id)
                H, W = frame.shape[:2]
                event_json["frame_w"] = W
                event_json["frame_h"] = H

                # 调试用打印输出
                print("\n=== 发送给前端的 JSON ===")
                print(event_json)

                send_json(event_json)  # 先 UDP发送给前端

                if need_report_to_node and curr_count > 0:
                    snap_dir: Path = cfg.SNAP_DIR
                    snap_dir.mkdir(exist_ok=True, parents=True)

                    ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(now_ts))
                    room_name = getattr(Config, "CAMERA_NAME", "exam_room1")
                    filename = f"{room_name}_{ts_str}_{curr_count}.jpg"
                    img_path = snap_dir / filename

                    vis_frame = _draw_boxes_on_frame(frame, all_boxes)

                    cv2.imwrite(str(img_path), vis_frame)
                    print("[Detection] 已保存截图:", img_path)

                    # 调用 reporter，把 event_json + imgPath 写库
                    report_exam_alarm(event_json, str(img_path))

                if curr_count > 0:
                    last_boxes_update_time = now_ts

                print(f"[AbnormalMsg] id={group_id}, count={curr_count}")

            prev_count = curr_count

def start_detection_thread(cfg: Config):
    # 在 run_server.py 里调用，起一个守护线程跑 _detection_loop
    t = threading.Thread(target=_detection_loop, args=(cfg,), daemon=True)
    t.start()
    print("[Main] Detection thread started.")
