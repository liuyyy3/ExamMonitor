# encoding: utf-8
# @File  : pose_infer.py
# @Author: Xinghui
# @Date  : 2025/12/5 15:20

# 封装 YOLO-pose.rknn 推理、解码关键点

import numpy as np
from rknn.api import RKNN

from utils.pose_decode import letterbox_resize, decode_yolov8_pose

class PoseInfer:
    def __init__(self,
                 model_path:str,
                 input_size=(640, 640), bg_color=56,
                 pose_conf_thr=0.3, nms_thr=0.4,
                 target = "rk3588"):
        self.model_path = model_path
        self.input_size = input_size
        self.bg_color = bg_color
        self.pose_conf_thr = pose_conf_thr
        self.nms_thr = nms_thr
        self.target = target

        self.rknn = RKNN(verbose=False)
        print(f"[PoseInfer]: Loading RKNN {model_path}")
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"Load pose rknn failed!")

        ret = self.rknn.init_runtime(target = target)
        if ret != 0:
            raise RuntimeError(f"Init pose runtime failed!")

    def infer(self, frame_bgr):
        letterbox_img, aspect_ratio, offset_x,offset_y = letterbox_resize(
            frame_bgr, self.input_size, self.bg_color)

        infer_img = letterbox_img[..., ::-1]
        infer_img = infer_img.transpose(2, 0, 1)
        infer_img = np.expand_dims(infer_img, 0).astype("uint8")
        outputs = self.rknn.inference(inputs=[infer_img], data_format="nchw")
        pose_results = decode_yolov8_pose(
            outputs,
            aspect_ratio = aspect_ratio,
            offset_x = offset_x,
            offset_y = offset_y,
        )

        return pose_results
