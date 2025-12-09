# encoding: utf-8
# @File  : json_schema.py
# @Author: Xinghui
# @Date  : 2025/12/5 15:25

# 统一封装 frame/event JSON 的生成

from datetime import datetime, timezone

from test import results


def _now_iso():
    # UTC ISO8601
    return datetime.now(timezone.utc).isoformat()


def make_abnormal_frame(boxes, group_id: int):
    ts = _now_iso()
    abnormal_count = len(boxes)

    msg = {
        "timestamp": ts,
        "abnormal_count": abnormal_count,
        "id": group_id,
    }

    if abnormal_count > 0:
        result_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box["bbox"]
            result_boxes.append({
                "type": box.get("type", "head_turn"),
                "x": int(x1),
                "y": int(y1),
                "w": int(x2 - x1),
                "h": int(y2 - y1),
                "color": box.get("color", "#ff0000"),  # 默认红色
            })
        msg["boxes"] = result_boxes

    return msg



