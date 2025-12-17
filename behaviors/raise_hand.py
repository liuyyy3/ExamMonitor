# encoding: utf-8
# @File  : raise_hand.py
# @Author: Xinghui
# @Date  : 2025/12/5 15:22

# 使用关键点做举手逻辑，输出【这一帧哪些框“举手”】

import math
import numpy as np

KP_CONF_TH = 0.35    # 关键点置信度阈值
ANGLE_TH = 55.0    # 前臂与竖直方向夹角阈值
DELTA_Y_RATIO = 0.08   # 肩-腕高度阈值系数
SIDE_RATIO_MAX = 1.2     # 水平偏移不超过肩宽的多少倍（防止侧伸）

def angle_between(v, u):
    vx, vy = v
    ux, uy = u
    dot = vx * ux + vy * uy
    nv = math.hypot(vx, vy)
    nu = math.hypot(ux, uy)
    if nv < 1e-6 or nu < 1e-6:
        return 180.0
    cosv = max(min(dot / (nv * nu), 1.0), -1.0)
    return math.degrees(math.acos(cosv))

def _hand_raise_one_side(kpts, side="left"):
    if side == "left":
        idx_sh, idx_el, idx_wr = 5, 7, 9
    else:
        idx_sh, idx_el, idx_wr = 6, 8, 10

    x_sh, y_sh, c_sh = kpts[idx_sh]
    x_el, y_el, c_el = kpts[idx_el]
    x_wr, y_wr, c_wr = kpts[idx_wr]

    # 关键点置信度不够直接否
    if c_sh < KP_CONF_TH or c_el < KP_CONF_TH or c_wr < KP_CONF_TH:
        return False

    # 身高估计：用肩到髋的距离
    x_lh, y_lh, c_lh = kpts[11]
    x_rh, y_rh, c_rh = kpts[12]
    if c_lh > KP_CONF_TH and c_rh > KP_CONF_TH:
        y_hip = (y_lh + y_rh) / 2.0
    else:
        y_hip = y_sh + 2.0 * abs(y_wr - y_sh)
    person_h = max(abs(y_hip - y_sh), 1.0)

    # 手腕要明显高于肘
    delta_y = DELTA_Y_RATIO * person_h
    if not (y_wr < y_el - delta_y):
        return False

    # 前臂基本竖直向上
    v_lower = (x_wr - x_el, y_wr - y_el)
    up = (0.0, -1.0)
    ang_lower = angle_between(v_lower, up)
    if ang_lower > ANGLE_TH:
        return False

    # 手不能伸到身体非常侧面
    x_Lsh, _, _ = kpts[5]
    x_Rsh, _, _ = kpts[6]
    shoulder_w = abs(x_Rsh - x_Lsh)
    if shoulder_w < 1e-3:
        return False
    if abs(x_wr - x_sh) > SIDE_RATIO_MAX * shoulder_w:
        return False

    # 手的大致高度要接近头部
    x_ns, y_ns, c_ns = kpts[0]   # nose
    if c_ns > KP_CONF_TH:
        if not (y_wr < y_ns + 0.15 * person_h):
            return False
    else:
        if not (y_wr < y_sh - 0.02 * person_h):
            return False

    return True

def is_raise_hand(kpts):
    left_up  = _hand_raise_one_side(kpts, "left")
    right_up = _hand_raise_one_side(kpts, "right")
    return left_up or right_up

def detect_raise_hands(
        frame_bgr,
        pose_results,
        kp_conf_thr = KP_CONF_TH,
):
    H, W  = frame_bgr.shape[:2]
    abnormal_boxes = []
    total_person = 0

    for det in pose_results:
        if det["class_id"] != 0:
            continue

        x1, y1, x2, y2 = det["bbox"]
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2))
        y2 = max(0, min(H - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        total_person += 1

        xy = det["kpt_xy"]
        conf = det["kpt_conf"]
        if len(xy) < 13:
            continue

        kpts = np.concatenate(
            [xy, conf.reshape(-1, 1)],
            axis=-1
        )

        if is_raise_hand(kpts):
            abnormal_boxes.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "label":"举手",
                "prob": 1.0,
                "type": "举手",
                "color": "#00ff00"  # 绿色
            })
    return abnormal_boxes, total_person