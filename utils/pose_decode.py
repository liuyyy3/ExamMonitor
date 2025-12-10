# encoding: utf-8
# @File  : pose_decode.py
# @Author: Xinghui
# @Date  : 2025/12/5 15:24

# letterbox、decode_yolov8_pose 等共用函数

import numpy as np
import cv2

POSE_CONF_THR = 0.3
NMS_THRESH = 0.4

CLASSES = ["person"]

nmsThresh = NMS_THRESH
objectThresh = POSE_CONF_THR

def letterbox_resize(image, size, bg_color):
    if isinstance(image, str):
        image = cv2.imread(image)

    target_width, target_height = size
    image_height, image_width, _ = image.shape

    aspect_ratio = min(target_width / image_width, target_height / image_height)
    new_width = int(image_width * aspect_ratio)
    new_height = int(image_height * aspect_ratio)

    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    result_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * bg_color
    offset_x = (target_width - new_width) // 2
    offset_y = (target_height - new_height) // 2
    result_image[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = image
    return result_image, aspect_ratio, offset_x, offset_y

class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax, keypoint):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.keypoint = keypoint

def IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    innerWidth = xmax - xmin
    innerHeight = ymax - ymin

    innerWidth = innerWidth if innerWidth > 0 else 0
    innerHeight = innerHeight if innerHeight > 0 else 0

    innerArea = innerWidth * innerHeight

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    total = area1 + area2 - innerArea
    if total <= 0:
        return 0.0
    return innerArea / total


def NMS(detectResult):
    predBoxs = []

    sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)

    for i in range(len(sort_detectboxs)):
        xmin1 = sort_detectboxs[i].xmin
        ymin1 = sort_detectboxs[i].ymin
        xmax1 = sort_detectboxs[i].xmax
        ymax1 = sort_detectboxs[i].ymax
        classId = sort_detectboxs[i].classId

        if sort_detectboxs[i].classId != -1:
            predBoxs.append(sort_detectboxs[i])
            for j in range(i + 1, len(sort_detectboxs), 1):
                if classId == sort_detectboxs[j].classId:
                    xmin2 = sort_detectboxs[j].xmin
                    ymin2 = sort_detectboxs[j].ymin
                    xmax2 = sort_detectboxs[j].xmax
                    ymax2 = sort_detectboxs[j].ymax
                    iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2)
                    if iou > nmsThresh:
                        sort_detectboxs[j].classId = -1
    return predBoxs


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax_np(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def process_yolo_level(out, keypoints, index, model_w, model_h, stride,
                       scale_w=1, scale_h=1):
    xywh = out[:, :64, :]
    conf = sigmoid(out[:, 64:, :])
    outputs = []
    for h in range(model_h):
        for w in range(model_w):
            for c in range(len(CLASSES)):
                if conf[0, c, (h * model_w) + w] > objectThresh:
                    xywh_ = xywh[0, :, (h * model_w) + w]
                    xywh_ = xywh_.reshape(1, 4, 16, 1)
                    data = np.array([i for i in range(16)]).reshape(1, 1, 16, 1)
                    xywh_ = softmax_np(xywh_, 2)
                    xywh_ = np.multiply(data, xywh_)
                    xywh_ = np.sum(xywh_, axis=2, keepdims=True).reshape(-1)

                    xywh_temp = xywh_.copy()
                    xywh_temp[0] = (w + 0.5) - xywh_[0]
                    xywh_temp[1] = (h + 0.5) - xywh_[1]
                    xywh_temp[2] = (w + 0.5) + xywh_[2]
                    xywh_temp[3] = (h + 0.5) + xywh_[3]

                    xywh_[0] = ((xywh_temp[0] + xywh_temp[2]) / 2)
                    xywh_[1] = ((xywh_temp[1] + xywh_temp[3]) / 2)
                    xywh_[2] = (xywh_temp[2] - xywh_temp[0])
                    xywh_[3] = (xywh_temp[3] - xywh_temp[1])
                    xywh_ = xywh_ * stride

                    xmin = (xywh_[0] - xywh_[2] / 2) * scale_w
                    ymin = (xywh_[1] - xywh_[3] / 2) * scale_h
                    xmax = (xywh_[0] + xywh_[2] / 2) * scale_w
                    ymax = (xywh_[1] + xywh_[3] / 2) * scale_h

                    keypoint = keypoints[..., (h * model_w) + w + index]
                    keypoint[..., 0:2] = keypoint[..., 0:2] // 1
                    box = DetectBox(c, conf[0, c, (h * model_w) + w],
                                    xmin, ymin, xmax, ymax, keypoint)
                    outputs.append(box)

    return outputs



def decode_yolov8_pose(outputs, aspect_ratio, offset_x, offset_y):
    keypoints_all = outputs[3]
    det_boxes = []
    for feat in outputs[:3]:
        index, stride = 0, 0
        if feat.shape[2] == 20:
            stride = 32
            index = 20 * 4 * 20 * 4 + 20 * 2 * 20 * 2
        if feat.shape[2] == 40:
            stride = 16
            index = 20 * 4 * 20 * 4
        if feat.shape[2] == 80:
            stride = 8
            index = 0
        feature = feat.reshape(1, 65, -1)
        det_boxes += process_yolo_level(
            feature, keypoints_all, index, feat.shape[3], feat.shape[2], stride
        )

    predbox = NMS(det_boxes)

    results = []
    for box in predbox:
        xmin = int((box.xmin - offset_x) / aspect_ratio)
        ymin = int((box.ymin - offset_y) / aspect_ratio)
        xmax = int((box.xmax - offset_x) / aspect_ratio)
        ymax = int((box.ymax - offset_y) / aspect_ratio)

        kpts = box.keypoint.reshape(-1, 3)
        kpts[..., 0] = (kpts[..., 0] - offset_x) / aspect_ratio
        kpts[..., 1] = (kpts[..., 1] - offset_y) / aspect_ratio

        results.append({
            'class_id': box.classId,
            'score':    float(box.score),
            'bbox':     (xmin, ymin, xmax, ymax),
            'kpt_xy':   kpts[:, :2],
            'kpt_conf': kpts[:, 2],
        })

    return results
