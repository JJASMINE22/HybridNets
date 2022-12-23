# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import os
import torch
import numpy as np
from PIL import Image
from torch.nn import functional as F
from hybridnet import HybridNet
from _utils.anchors import Anchors
from _utils.generate import Generator
from configure import config as cfg

if __name__ == '__main__':

    priors = Anchors(scales=cfg.scales,
                     ratios=cfg.ratios)(cfg.input_size)

    Hybridnet = HybridNet(fpn_cells=cfg.fpn_cells,
                          num_layers=cfg.num_layers,
                          num_anchors=cfg.num_anchors,
                          num_classes=cfg.class_names.__len__() + 1,
                          seg_classes=cfg.segmentation_class_names.__len__(),
                          num_features=cfg.num_features,
                          conv_channels=cfg.conv_channels,
                          out_indices=cfg.out_indices,
                          up_scale=cfg.up_scale,
                          backbone=cfg.backbone,
                          priors=priors,
                          learning_rate=cfg.learning_rate,
                          weight_decay=cfg.weight_decay,
                          iou_thresh=cfg.iou_thresh,
                          nms_thresh=cfg.nms_thresh,
                          resume_train=cfg.resume_train,
                          ckpt_path=cfg.ckpt_path + "\\模型文件")

    Hybridnet.model = Hybridnet.model.eval()

    while True:
        file_path = input("Enter file path: ")
        try:
            source = Image.open(file_path)
            source = source.resize(reversed(cfg.input_size))
            source = np.array(source, dtype="float32")
            source = source / 255.
            source = source.transpose([2, 0, 1])
            sources = source[np.newaxis]
            Hybridnet.generate_sample(sources, 0)
        except Exception as e:
            print(repr(e))