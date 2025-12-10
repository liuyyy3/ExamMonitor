# encoding: utf-8
# @File  : reporter.py
# @Author: Xinghui
# @Date  : 2025/12/10 13:05

# 把结果 POST 给 Node

import requests, json, time
from pathlib import Path
from app.config import Config

def report_exam_alarm(event_json: dict, img_path: str):
    if event_json is None:
        event_json = {}

    payload = dict(event_json)
    payload["MediaUrl"] = getattr(Config, "RTSP_URL", "")
    payload["imgPath"] = str(img_path or "")

    # 调试用打印输出
    print("\n=== POST 给 Node 的 JSON ===")
    print(payload)

    headers = {}
    if Config.TOKEN:
        headers["Authorization"] = f"Bearer {Config.TOKEN}"

    try:
        r = requests.post(
            Config.REPORT_URL,
            json=payload,
            headers=headers,
            timeout=8,
        )
        r.raise_for_status()
        print("[ExamReporter] 上报成功:", r.text[:200])
    except Exception as e:
        print("[ExamReporter] 上报失败:", e)

