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

import numpy as np
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None

from app.config import Config
from models.pose_infer import PoseInfer
from models.headturn_cls import HeadTurnClassifier
from behaviors.head_turn import detect_head_turns
from behaviors.raise_hand import detect_raise_hands

from utils.json_schema import make_abnormal_frame
from socket_sever.udp_multicast import send_json
from utils.reporter import report_exam_alarm

from socket_sever.rtsp_state import get_rtsp_url

# 事件队列：routes.py 从这里取事件
_event_queue: queue.Queue = queue.Queue()

# 当前状态：给 /status 用
_current_abnormal = False
_current_boxes = []  # 当前异常框（list[dict]），结构和 event 里的 boxes 相同

_FONT_PATH_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttf",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_font_cache = {}


def _get_label_font(font_size=18):
    if ImageFont is None:
        return None
    if font_size in _font_cache:
        return _font_cache[font_size]
    for path in _FONT_PATH_CANDIDATES:
        p = Path(path)
        if p.exists():
            try:
                font = ImageFont.truetype(str(p), size=font_size)
                _font_cache[font_size] = font
                return font
            except OSError:
                continue

    try:
        font = ImageFont.load_default()
        _font_cache[font_size] = font
        return font
    except Exception:
        return None

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

def _draw_boxes_on_frame(frame, boxes, font_size=18, box_thickness=2):
    # 在图像上画出 all_boxes 里的框，颜色来自 box['color']
    vis = frame.copy()

    font_size = max(1, int(font_size or 18))
    box_thickness = max(1, int(box_thickness or 2))

    pil_image = None
    draw = None
    font = _get_label_font(font_size)
    if Image is not None and font is not None:
        pil_image = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]
        color_str = box.get("color", "#ff0000")
        color = _hex_to_bgr(color_str)

        if draw is not None:
            outline_color = (color[2], color[1], color[0])
            draw.rectangle(
                [(int(x1), int(y1)), (int(x2), int(y2))],
                outline=outline_color,
                width=box_thickness,
            )
        else:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, box_thickness)

        # 如果想在图上写文字，可以顺便把 label 也写上：
        label = box.get("type", "")
        # if label:
        if not label:
            continue
        if draw is not None:
            text = str(label)
            try:
                text_box = draw.textbbox((0, 0), text, font=font)
                text_h = text_box[3] - text_box[1]
            except Exception:
                # 老版 Pillow
                _, text_h = draw.textsize(text, font=font)

            text_x = int(x1)
            text_y = max(0, int(y1) - text_h - 4)
            draw.text((text_x, text_y), text, font=font, fill=outline_color)
        else:
            # 回退到 OpenCV 内置字体（不支持中文，但至少不中断流程）
            font_scale = max(0.5, font_size / 30.0)
            cv2.putText(
                vis, label,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,  color, max(1, box_thickness), cv2.LINE_AA
            )

    if pil_image is not None:
        vis = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
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


class BoxSmoother:
    # raw_boxes: list[dict]  每个 dict 至少包含: type,x,y,w,h,color
    # 输出: 平滑后的 list[dict]

    def __init__(self, miss_max=2, ema_alpha=0.6, match_iou=0.2):
        self.miss_max = miss_max
        self.ema_alpha = ema_alpha
        self.match_iou = match_iou
        self.tracks = {}   # tid -> track
        self._next_id = 1

    def _new_track(self, b, now_ts):
        tid = self._next_id
        self._next_id += 1
        self.tracks[tid] = {
            "type": b.get("type", ""),
            "color": b.get("color", "#ff0000"),
            "bbox": [float(v) for v in b["bbox"]],
            "miss": 0,
            "last_ts": now_ts,
        }
        return tid

    def update(self, raw_boxes, now_ts):
        matched_tids = set()

        # 1) 逐个 raw box 去匹配老 track（同 type + IoU 最大）
        for b in raw_boxes:
            if "bbox" not in b:
                continue
            b_type = b.get("type", "")
            b_bbox = [float(v) for v in b["bbox"]]

            best_tid = None
            best_iou = 0.0
            for tid, tr in self.tracks.items():
                if tr["type"] != b_type:
                    continue
                if tid in matched_tids:
                    continue
                iou = _iou(b_bbox, tr["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None and best_iou >= self.match_iou:
                # 2) 匹配上：EMA 更新 bbox
                tr = self.tracks[best_tid]
                a = self.ema_alpha
                old = tr["bbox"]
                new = b_bbox
                tr["bbox"] = [
                    a * new[0] + (1 - a) * old[0],
                    a * new[1] + (1 - a) * old[1],
                    a * new[2] + (1 - a) * old[2],
                    a * new[3] + (1 - a) * old[3],
                ]
                tr["miss"] = 0
                tr["last_ts"] = now_ts
                tr["color"] = b.get("color", tr["color"])
                matched_tids.add(best_tid)
            else:
                # 3) 没匹配上：新建 track
                tid = self._new_track(b, now_ts)
                matched_tids.add(tid)

        # 4) 没被匹配到的 track：miss += 1，超过阈值删除
        to_del = []
        for tid, tr in self.tracks.items():
            if tid in matched_tids:
                continue
            tr["miss"] += 1
            if tr["miss"] > self.miss_max:
                to_del.append(tid)
        for tid in to_del:
            self.tracks.pop(tid, None)

        # 5) 输出平滑后的 boxes（保持内部 bbox schema）
        out = []
        for tid, tr in self.tracks.items():
            out.append({
                "type": tr["type"],
                "color": tr["color"],
                "bbox": tr["bbox"],
                # 可选：带上 tid，未来前端如果愿意可用它更稳
                "tid": tid,
            })
        return out


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

# 新增一个函数功能，当框的移动过大的时候，就不给前端发送 json
def _boxes_move_too_much(prev_boxes, curr_boxes, max_move_px):
    if max_move_px <= 0:
        return False
    if len(prev_boxes) == 0 or len(curr_boxes) == 0:
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

        if best_idx == -1:
            return True

        used[best_idx] = True

        cb = curr_boxes[best_idx]["bbox"]
        px = (pb[0] + pb[2]) / 2.0
        py = (pb[1] + pb[3]) / 2.0
        cx = (cb[0] + cb[2]) / 2.0
        cy = (cb[1] + cb[3]) / 2.0
        if np.hypot(cx - px, cy - py) > max_move_px:
            return True

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
    # routes 调用：返回当前状态（只读）
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

    smoother = BoxSmoother(
        miss_max=getattr(cfg, "BOX_MISS_MAX", 2),
        ema_alpha=getattr(cfg, "BOX_EMA_ALPHA", 0.6),
        match_iou=getattr(cfg, "BOX_MATCH_IOU", 0.2),
    )

    frame_idx = 0

    group_id = 0  # 当前异常人群 id（自增）
    prev_boxes_for_group = []  # 上一轮用于对比分组的 boxes
    prev_count = 0  # 上一轮异常人数
    last_boxes_update_time = 0.0  # 上一次“带 boxes 的消息”时间（仅在 count>0 时更新）

    last_infer_ms = 0
    cached_event_json = None  # 用于显示最近消息

    cap = None
    current_url = None

    while True:
        desired_url = get_rtsp_url() or getattr(cfg, "RTSP_URL", None)
        if not desired_url:
            time.sleep(0.2)
            continue

        cap = cv2.VideoCapture(desired_url)
        current_url = desired_url
        if not cap.isOpened():
            print(f"[DetectionLoop] RTSP 打不开，{cfg.RECONNECT_INTERVAL_SEC}s 后重试: {desired_url}")
            time.sleep(cfg.RECONNECT_INTERVAL_SEC)
            continue

        while True:
            desired_url = get_rtsp_url() or getattr(cfg, "RTSP_URL", None)
            if not desired_url:
                time.sleep(0.2)
                break

            if cap is None or desired_url != current_url:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass

                current_url = desired_url
                cap = cv2.VideoCapture(current_url)
                if not cap.isOpened():
                    print(f"[RTSP] switch failed, {cfg.RECONNECT_INTERVAL_SEC}s retry: {current_url}")
                    time.sleep(cfg.RECONNECT_INTERVAL_SEC)
                    continue
                print(f"[RTSP] switch to: {current_url}")

                # 切换流后重置一些计时器，避免立刻触发定时发送
                last_infer_ms = 0
                last_boxes_update_time = 0.0

            ret, frame = cap.read()
            if not ret:
                print("[DetectionLoop] 读取帧失败，准备重连 RTSP ...")
                cap.release()
                cap = None
                time.sleep(cfg.RECONNECT_INTERVAL_SEC)
                break  # 跳出内层 while，重新走外层，重连 RTSP

            now_ms = int(time.time() * 1000)
            interval = int(getattr(cfg, "INFER_INTERVAL_MS", 1))

            if interval > 0 and (now_ms - last_infer_ms) < interval:
                continue
            last_infer_ms = now_ms

            frame_idx += 1
            # # 跳帧检测
            # if cfg.FRAME_STRIDE > 1 and (frame_idx % cfg.FRAME_STRIDE != 0):
            #     continue

            # YOLO-pose 检测（只返回 person + 关键点）
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

            raw_boxes = head_boxes + raise_boxes
            now_ts = time.time()

            all_boxes = smoother.update(raw_boxes, now_ts)
            curr_count = len(all_boxes)


            # 更新当前状态，给 /status 用
            _update_state(
                abnormal = (len(all_boxes) > 0),
                boxes = all_boxes,
            )

            # curr_count==0 且没变化时不会定时发，只会在从有异常→0 的那一刻发一次
            now_ts = time.time()
            send_msg = False
            need_report_to_node = False

            group_change = False

            # 向前端发送 json的规则
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

            if curr_count > 0 and(not group_change):
                prev_boxes_for_group = [b.copy() for b in all_boxes]

            if curr_count != prev_count:
                send_msg = True
            elif curr_count > 0 and (now_ts - last_boxes_update_time >= cfg.FRAME_MSG_INTERVAL):
                send_msg = True

            if send_msg:
                # event_json = make_abnormal_frame(all_boxes, group_id)

                if _boxes_move_too_much(
                    prev_boxes_for_group,
                    all_boxes,
                    getattr(cfg, "BOX_MOVE_MAX_PX", 0),
                ):
                    print("[AbnormalMsg] skip send: boxes move too much")
                    send_msg = False
                    need_report_to_node = False
                else:
                    event_json = make_abnormal_frame(all_boxes, group_id)

                H, W = frame.shape[:2]
                if send_msg:
                    event_json["frame_w"] = W
                    event_json["frame_h"] = H


                    # 调试用打印输出
                    print("\n=== 发送给前端的 JSON ===")
                    print(event_json)

                    send_json(event_json)  # 先 UDP发送给前端

                    if need_report_to_node and curr_count > 0:
                        snap_dir: Path = cfg.SNAP_DIR
                        snap_dir.mkdir(exist_ok=True, parents=True)

                        # 保存图片时候的命名规则
                        ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(now_ts))
                        room_name = getattr(cfg, "CAMERA_NAME", "exam_room1")
                        filename = f"{room_name}_{ts_str}_{curr_count}.jpg"
                        img_path = snap_dir / filename

                    # vis_frame = _draw_boxes_on_frame(frame, all_boxes)
                        vis_frame = _draw_boxes_on_frame(
                            frame,
                            all_boxes,
                            font_size=getattr(cfg, "LABEL_FONT_SIZE", 18),
                            box_thickness=getattr(cfg, "BOX_THICKNESS", 2),
                        )

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
