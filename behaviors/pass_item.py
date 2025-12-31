# encoding: utf-8
# @File  : pass_item.py
# @Author: Xinghui
# @Date  : 2025/12/30 15:32

# 新增检测传递物品的功能, 暂时使用关键点做疑似传递物品逻辑，输出相应的 json给到前端

import numpy as np

KP_CONF_TH = 0.25
# 肩/肘/腕 y 允许的最大偏差（相对身高）
HORIZ_Y_RATIO = 0.18  # 非常规整、正面、实验室0.08 ~ 0.10， 普通考试监控（推荐）0.12 ~ 0.18， 俯视角明显 / 桌椅不齐0.20 ~ 0.25
MIN_Y_TOL = 12.0


def _estimate_person_height(kpts, kp_conf_thr=KP_CONF_TH):
    x_sh_l, y_sh_l, c_sh_l = kpts[5]
    x_sh_r, y_sh_r, c_sh_r = kpts[6]
    y_sh = None
    if c_sh_l > kp_conf_thr and c_sh_r > kp_conf_thr:
        y_sh = (y_sh_l + y_sh_r) / 2.0
    elif c_sh_l > kp_conf_thr:
        y_sh = y_sh_l
    elif c_sh_r > kp_conf_thr:
        y_sh = y_sh_r

    x_lh, y_lh, c_lh = kpts[11]
    x_rh, y_rh, c_rh = kpts[12]
    y_hip = None
    if c_lh > kp_conf_thr and c_rh > kp_conf_thr:
        y_hip = (y_lh + y_rh) / 2.0
    elif c_lh > kp_conf_thr:
        y_hip = y_lh
    elif c_rh > kp_conf_thr:
        y_hip = y_rh

    if y_sh is None or y_hip is None:
        return 0.0
    return abs(y_hip - y_sh)


def _is_arm_horizontal(kpts, side="left", kp_conf_thr=KP_CONF_TH):
    if side == "left":
        idx_sh, idx_el, idx_wr = 5, 7, 9
    else:
        idx_sh, idx_el, idx_wr = 6, 8, 10

    x_sh, y_sh, c_sh = kpts[idx_sh]
    x_el, y_el, c_el = kpts[idx_el]
    x_wr, y_wr, c_wr = kpts[idx_wr]

    if c_sh < kp_conf_thr or c_el < kp_conf_thr or c_wr < kp_conf_thr:
        return False

    person_h = _estimate_person_height(kpts, kp_conf_thr=kp_conf_thr)
    if person_h <= 0:
        return False

    y_tol = max(MIN_Y_TOL, HORIZ_Y_RATIO * person_h)
    if y_el < y_tol or y_wr < y_sh:
        return False
    if max(y_el - y_sh, y_wr - y_sh) > y_tol:
        return False

    return True


def is_pass_item(kpts):
    left_ok = _is_arm_horizontal(kpts, "left")
    right_ok = _is_arm_horizontal(kpts, "right")
    return left_ok or right_ok


def detect_pass_items(
    frame_bgr,
    pose_results,
    kp_conf_thr=KP_CONF_TH,
):
    H, W = frame_bgr.shape[:2]
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

        if is_pass_item(kpts):
            abnormal_boxes.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "label": "传递可疑物",
                "prob": 1.0,
                "type": "传递可疑物",
                "color": "#7b61ff"
            })

    return abnormal_boxes, total_person


