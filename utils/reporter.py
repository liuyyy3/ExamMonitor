# encoding: utf-8
# @File  : reporter.py
# @Author: Xinghui
# @Date  : 2025/12/10 13:05

# 把结果 POST 给 Node,存入数据库

import requests, json, time
from pathlib import Path
from app.config import Config
from datetime import datetime, timezone, timedelta
from utils.logging_util import get_logger

def report_exam_alarm(event_json: dict, img_path: str):
    logger = get_logger()

    if event_json is None:
        event_json = {}

    # 从发给前端的 boxes中去除type
    type_set = set()
    for b in event_json.get("boxes", []) or []:
        t = b.get("type")
        if t:
            type_set.add(t)
    type_list = sorted(type_set)

    # 新增一个北京时间的字段
    utc_ts_str = event_json.get("timestamp", "")
    timestamp_cn = ""

    if utc_ts_str:
        dt_utc = datetime.fromisoformat(utc_ts_str.replace("Z", "+00:00"))
        dt_cn = dt_utc.astimezone(timezone(timedelta(hours=8)))
        timestamp_cn = dt_cn.strftime("%Y-%m-%d %H:%M:%S")

    payload = {
        "timestamp": event_json.get("timestamp", ""),
        "AbnormalCount": event_json.get("abnormal_count", 0),

        "id": event_json.get("id", 0),
        "MediaName":"1",
        "MediaUrl": getattr(Config, "RTSP_URL", ""),
        "ResultType": type_list,
        "ResultDescription":"",
        "imgPath": str(img_path or ""),
        "UploadReason": "",
        "Uploadstatus": "1",
        "Uploadvideo_path":"",
        "UserData": {},
        "created_at": timestamp_cn,
    }

    # 调试用打印输出
    print("\n=== POST 给 Node 的 JSON ===")
    logger.info("\n=== POST 给 Node 的 JSON ===")
    print(payload)
    logger.info(payload)

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
        logger.info("[ExamReporter] 上报成功:", r.text[:200])
    except Exception as e:
        print("[ExamReporter] 上报失败:", e)
        logger.info("[ExamReporter] 上报失败:", e)

