# encoding: utf-8
# @File  : head_turn.py
# @Author: Xinghui
# @Date  : 2025/12/5 15:21

# 使用 pose + mobilenet 输出【这一帧哪些框是“转头”】

import numpy as np
import cv2

from models.headturn_cls import HeadTurnClassifier

def draw_single_person_skeleton(h, w, xy, conf, conf_thr=0.3):
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    nose_idx, leye_idx, reye_idx, lear_idx, rear_idx = 0, 1, 2, 3, 4
    lshoulder_idx, rshoulder_idx = 5, 6
    lelbow_idx, relbow_idx = 7, 8
    lwrist_idx, rwrist_idx = 9, 10
    lhip_idx, rhip_idx = 11, 12

    torso_arm_pairs = [
        (lshoulder_idx, rshoulder_idx),
        (lshoulder_idx, lhip_idx),
        (rshoulder_idx, rhip_idx),
        (lhip_idx, rhip_idx),
        (lshoulder_idx, lelbow_idx),
        (lelbow_idx, lwrist_idx),
        (rshoulder_idx, relbow_idx),
        (relbow_idx, rwrist_idx),
    ]

    neck_valid = False
    neck_x = neck_y = 0.0
    if conf[lshoulder_idx] > conf_thr and conf[rshoulder_idx] > conf_thr:
        lx, ly = xy[lshoulder_idx]
        rx, ry = xy[rshoulder_idx]
        neck_x = (lx + rx) / 2.0
        neck_y = (ly + ry) / 2.0
        neck_valid = True

    for j in range(13):
        if conf[j] < conf_thr:
            continue
        x, y = int(xy[j, 0]), int(xy[j, 1])
        cv2.circle(canvas, (x, y), 3, (0, 255, 0), -1)

    if neck_valid:
        cv2.circle(canvas, (int(neck_x), int(neck_y)), 4, (0, 255, 255), -1)

    for a, b in torso_arm_pairs:
        if conf[a] < conf_thr or conf[b] < conf_thr:
            continue
        x1, y1 = int(xy[a, 0]), int(xy[a, 1])
        x2, y2 = int(xy[b, 0]), int(xy[b, 1])
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

    nose_ok = conf[nose_idx] > conf_thr
    leye_ok = conf[leye_idx] > conf_thr
    reye_ok = conf[reye_idx] > conf_thr

    if nose_ok or leye_ok or reye_ok:
        if neck_valid and nose_ok:
            nx, ny = int(xy[nose_idx, 0]), int(xy[nose_idx, 1])
            cv2.line(canvas, (int(neck_x), int(neck_y)), (nx, ny), (0, 0, 255), 2)

        if leye_ok and nose_ok:
            lx, ly = int(xy[leye_idx, 0]), int(xy[leye_idx, 1])
            nx, ny = int(xy[nose_idx, 0]), int(xy[nose_idx, 1])
            cv2.line(canvas, (lx, ly), (nx, ny), (255, 0, 0), 2)

        if reye_ok and nose_ok:
            rx, ry = int(xy[reye_idx, 0]), int(xy[reye_idx, 1])
            nx, ny = int(xy[nose_idx, 0]), int(xy[nose_idx, 1])
            cv2.line(canvas, (rx, ry), (nx, ny), (255, 0, 0), 2)

        if conf[lear_idx] > conf_thr and leye_ok:
            ex, ey = int(xy[lear_idx, 0]), int(xy[lear_idx, 1])
            lx, ly = int(xy[leye_idx, 0]), int(xy[leye_idx, 1])
            cv2.line(canvas, (ex, ey), (lx, ly), (255, 0, 255), 2)

        if conf[rear_idx] > conf_thr and reye_ok:
            ex, ey = int(xy[rear_idx, 0]), int(xy[rear_idx, 1])
            rx, ry = int(xy[reye_idx, 0]), int(xy[reye_idx, 1])
            cv2.line(canvas, (ex, ey), (rx, ry), (255, 0, 255), 2)
    else:
        lear_idx2, rear_idx2 = lear_idx, rear_idx
        lear_ok = conf[lear_idx2] > conf_thr
        rear_ok = conf[rear_idx2] > conf_thr

        head_valid = False
        head_x = head_y = 0.0

        if lear_ok and rear_ok:
            lx, ly = xy[lear_idx2]
            rx, ry = xy[rear_idx2]
            head_x = (lx + rx) / 2.0
            head_y = (ly + ry) / 2.0
            head_valid = True
        elif lear_ok and neck_valid:
            lx, ly = xy[lear_idx2]
            head_x = (lx + neck_x) / 2.0
            head_y = (ly + neck_y) / 2.0
            head_valid = True
        elif rear_ok and neck_valid:
            rx, ry = xy[rear_idx2]
            head_x = (rx + neck_x) / 2.0
            head_y = (ry + neck_y) / 2.0
            head_valid = True
        elif neck_valid:
            head_x, head_y = neck_x, neck_y
            head_valid = True

        if head_valid:
            hx, hy = int(head_x), int(head_y)
            cv2.circle(canvas, (hx, hy), 5, (0, 165, 255), -1)
            if neck_valid:
                cv2.line(canvas, (int(neck_x), int(neck_y)), (hx, hy),
                         (0, 165, 255), 2)
            if lear_ok:
                ex, ey = int(xy[lear_idx2, 0]), int(xy[lear_idx2, 1])
                cv2.line(canvas, (ex, ey), (hx, hy), (255, 0, 255), 2)
            if rear_ok:
                ex, ey = int(xy[rear_idx2, 0]), int(xy[rear_idx2, 1])
                cv2.line(canvas, (ex, ey), (hx, hy), (255, 0, 255), 2)

    return canvas

def detect_head_turns(
    frame_bgr,
    pose_results,
    cls_model: HeadTurnClassifier,
    kp_conf_thr=0.2,
    min_valid_kpts=3,
    head_turn_prob_thr=0.5,
):
    H, W = frame_bgr.shape[:2]

    abnormal_boxes = []
    total_person = []

    for det in pose_results:
        if det['class_id'] != 0:
            continue
        x1, y1, x2, y2 = det['bbox']
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2))
        y2 = max(0, min(H - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        total_person += 1
        
        xy = det['kpt_xy']
        conf = det['kpt_conf']

        if int((conf[:13] > kp_conf_thr).sum()) < min_valid_kpts:
            continue

        w_box = x2 - x1
        h_box = y2 - y1
        if w_box < 20 or h_box < 20:
            continue

        xy_local = xy.copy()
        xy_local[:, 0] -= x1
        xy_local[:, 1] -= y1

        skel_canvas = draw_single_person_skeleton(
            h_box, w_box, xy_local, conf, conf_thr=kp_conf_thr
        )

        label, prob = cls_model.predict_single(skel_canvas)

        if label == "turn_head" and prob >= head_turn_prob_thr:
            abnormal_boxes.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "label": label,
                "prob": float(prob),
                "type": "head_turn",
                "color": '#ff0000'  # 红色
            })

    return abnormal_boxes, total_person
