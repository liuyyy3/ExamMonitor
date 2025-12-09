# encoding: utf-8
# @File  : routes.py
# @Author: Xinghui
# @Date  : 2025/12/5 15:15

# 对 Qt 暴露的 HTTP 接口 (/api/frame, /api/events)

from flask import Blueprint, jsonify
from app.detection_loop import pop_all_events, get_current_state

bp = Blueprint('api', __name__)

@bp.route("ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})

@bp.route("/events", methods=["GET"])
def events():
    events = pop_all_events()
    return jsonify({"events": events})

@bp.route("/state", methods=["GET"])
def state():
    state = get_current_state()
    return jsonify(state)


