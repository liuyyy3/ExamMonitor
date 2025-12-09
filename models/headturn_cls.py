# encoding: utf-8
# @File  : headturn_cls.py
# @Author: Xinghui
# @Date  : 2025/12/5 15:21

# 封装 MobileNet.rknn 分类推理

import numpy as np
import cv2
from rknn.api import RKNN

# 训练时用过的归一化参数
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SKELETON_SIZE = 224  # 输入大小

def softmax_np(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

class HeadTurnClassifier:
    def __init__(self, model_path: str, class_names):
        self.model_path = model_path
        self.class_names = class_names

        self.rknn = RKNN(verbose=False)
        print(f"[HeadTurnClassifier] Load RKNN: {model_path}")
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError("Load cls rknn failed!")

        ret = self.rknn.init_runtime(target="rk3588")
        if ret != 0:
            raise RuntimeError("Init cls runtime failed!")


    @staticmethod
    def preprocess_skeleton_bgr(img_bgr):
        """
        对骨架图做和训练完全一致的预处理：
        BGR -> RGB -> resize(224,224) -> /255 -> Normalize -> NCHW
        """
        img_bgr = cv2.resize(img_bgr, (SKELETON_SIZE, SKELETON_SIZE))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        img = img_rgb.astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        img = np.transpose(img, (2, 0, 1))  # (3,224,224)
        img = np.expand_dims(img, 0)  # (1,3,224,224)

        return img

    def predict_single(self, skel_bgr):
        inp = self.preprocess_skeleton_bgr(skel_bgr)
        out = self.rknn.inference(inputs=[inp], data_format="nchw")[0]
        prob = softmax_np(out, axis=0)[0]
        idx = int(np.argmax(prob))
        label = self.class_names[idx]
        pmax = float(prob[idx])
        return label, pmax


