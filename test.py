# encoding: utf-8
# @File  : test.py
# @Author: Xinghui
# @Date  : 2025/12/4 9:38

# 车市一下如果只是单单跑官网下载来的 yolo-pose是会检测出几个关键点

from ultralytics import YOLO
from PIL import Image

model = YOLO(r"D:\PyProject\exam_cheat_yolo_cls\models\yolo11m-pose.pt")
results = model(r"D:\Datasets\Exam\bnd1\bnd1_00h26m53s.jpg")

for i, r in enumerate(results):
    im_bgr = r.plot()
    im_rgb = Image.fromarray(im_bgr[..., ::-1])

    r.show()





