from __future__ import division
from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import argparse
import cv2
import torch
from torch.autograd import Variable

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default='/content/yolo-movie-class/src/input.jpg', help="path to image file")
#    parser.add_argument("--image_path", type=str, default='/content/yolo-movie-class/src/shrimp_target.jpg', help="path to image file")
#   parser.add_argument("--image_path", type=str, default='/content/yolo-movie-class/JPEGImages/IMG_11_jpg.rf.c9c10e73633fb21009816078bc689ba6.jpg', help="path to image file")
#    parser.add_argument("--image_path", type=str, default='/content/yolo-movie-class/JPEGImages/00032.jpg', help="path to image file")
    parser.add_argument("--model_def", type=str, default="/content/yolo-movie-class/src/config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="/content/yolo-movie-class/src/weights/yolov3_ckpt_200.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="/content/yolo-movie-class/src/data/custom/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.2, help="iou threshold for non-maximum suppression")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    model.load_state_dict(torch.load(opt.weights_path))
    model.eval()  # Set in evaluation mode

    classes = load_classes(opt.class_path)  # Extract class labels from file
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # Load image
    img_path = opt.image_path
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape

    x = torch.from_numpy(img.transpose(2, 0, 1))
    x = x.unsqueeze(0).float()  # x = (1, 3, H, W)

    # Apply letterbox resize
    _, _, h, w = x.size()
    ih, iw = (416, 416)
    dim_diff = abs(h - w)
    pad1, pad2 = int(dim_diff // 2), int(dim_diff - dim_diff // 2)
    pad = (pad1, pad2, 0, 0) if w <= h else (0, 0, pad1, pad2)
    x = F.pad(x, pad=pad, mode='constant', value=127.5) / 255.0
    x = F.upsample(x, size=(ih, iw), mode='bilinear')  # x = (1, 3, 416, 416)
    x = x.to(device)

    dynamic_conf_thres = 0.01  # ここを調整してください
    with torch.no_grad():
        detections = model(x)
        detections = non_max_suppression(detections, dynamic_conf_thres, opt.nms_thres)  # 動的に設定したスコアを使用

    detections = detections[0]
    if detections is not None:
        detections = rescale_boxes(detections, opt.img_size, [height, width])
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('/content/output.jpg', img)  # 画像を保存
    length= len(detections)-1
    print("個数:", length)

