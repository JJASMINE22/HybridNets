import os
import torch
import numpy as np
from _utils.utils import get_classes, get_segmentation_classes


# ===annotation===
annotation_path = "标签/注释根目录"
image_root = "数据根目录"

# ===generator===
classes_path = "放置于/model_data下记录检测类别的文件classes.txt"
segmentation_classes_path = "放置于/model_data下记录图像分割种类的文件segmentation_classes.txt"
batch_size = 4
train_split = 0.6
input_size = (384, 640)
scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]

# ===model===
backbone = "efficientnet_b0" # 骨干网络
fpn_cells = 6
num_layers = 4
num_anchors = scales.__len__() * ratios.__len__()
num_features = 160
conv_channels = [24, 40, 112, 320]
up_scale = (4, 4)
out_indices = (1, 2, 3, 4)

# ===training===
Epoches = 100
resume_train = True
learning_rate = 5e-4
weight_decay = 5e-4
class_names = get_classes(classes_path)
segmentation_class_names = get_segmentation_classes(segmentation_classes_path)
device = torch.device('cuda') if torch.cuda.is_available() else None
ckpt_path = "模型存储根目录, 放置于/checkpoint下"

# ===prediction===
iou_thresh = .5
nms_thresh = .3
font_color = (0, 255, 255)
rect_color = (0, 0, 255)
thickness = .5
per_sample_interval = 100
segmentation_colors = np.array([[0, 0, 0],
                                [255, 0, 0],
                                [0, 255, 0]])  # adjustable
font_path = "字体根目录, 放置于/font下"
sample_path = "训练生成的图像根目录, 放置于/result/Batch{}.jpg"
