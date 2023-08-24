#!/usr/bin/env python3

import rospy
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import torch

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = current_dir + '/yolov5'
print(current_dir)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.general import LOGGER, check_img_size, cv2, non_max_suppression, scale_boxes, xyxy2xywh
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox
 
class Yolo:
    def __init__(self, model_path=ROOT/'crash.pt',
                    imgsz=(320, 320),
                    conf_thres=0.25,
                    iou_thres=0.45,
                    max_det=20,
                    device='cpu',
                    ):

        self.weights = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = select_device(device)
        self.classes = None
        self.agnostic_nms = False
        self.augment = False
        self.visualize = False
        self.half = False
        self.dnn = False

        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn)
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        self.imgsz = check_img_size(imgsz, s=self.stride)
 
    def detect(self, img: cv2.Mat):
        self.model.warmup(imgsz=(1, 3, *self.imgsz))
        dt, seen = [0.0, 0.0, 0.0], 0

        im0 = img
        im = letterbox(im0, self.imgsz, self.stride, auto=self.pt)[0]
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)
        t1 = time_sync()
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        t2 = time_sync()
        dt[0] += t2 - t1

        pred = self.model(im, augment=self.augment, visualize=self.visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        dt[2] += time_sync() - t3

        detections=[]

        for i, det in enumerate(pred):
            seen += 1
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                    xywh = [round(x) for x in xywh]
                    xywh = [xywh[0] - xywh[2] // 2, xywh[1] - xywh[3] // 2, xywh[2], xywh[3]]
                    x = xywh[0]
                    y = xywh[1]
                    w = xywh[2]
                    h = xywh[3]
    
                    detections.append([[[x, y], [x + w, y], [x + w, y + h], [x, y + h]], int(cls)])

        LOGGER.info(f'({t3 - t2:.3f}s)')
        rospy.loginfo(detections)
        return detections
    

if __name__ == "__main__":
    path = '/home/shine/yolov5/datasets/digit/images/train/0305.jpg'
    img = cv2.imread(path)
    cv2.imshow("src", img)
    yolo = Yolo()
    img_labels = yolo.detect(img)
    for label in img_labels:
        print(label)
        img_dst = cv2.rectangle(img, label[0][0], label[0][2], (0, 255, 0), 2)
        cv2.imshow("dst", img_dst)
        cv2.waitKey(0)
