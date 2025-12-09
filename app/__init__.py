# encoding: utf-8
# @File  : __init__.py
# @Author: Xinghui
# @Date  : 2025/12/5 15:15

# 创建 Flask 对象，注册蓝图

from flask import Flask
from .routes import bp as api_bp

def create_app():
    app = Flask(__name__)
    app.register_blueprint(api_bp, url_prefix='/api')
    return app