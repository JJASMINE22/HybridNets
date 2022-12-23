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

    data_gen = Generator(image_root=cfg.image_root,
                         anno_path=cfg.annotation_path,
                         input_size=cfg.input_size,
                         batch_size=cfg.batch_size,
                         train_split=cfg.train_split,
                         priors=priors,
                         num_classes=cfg.class_names.__len__())

    train_gen = data_gen.generate(training=True)
    validate_gen = data_gen.generate(training=False)

    for epoch in range(cfg.Epoches):
        for i in range(data_gen.get_train_len()):
            print(i+1)
            sources, sg_sources, targets = next(train_gen)
            Hybridnet.train(sources, sg_sources, targets)
            if not (i + 1) % cfg.per_sample_interval:
                Hybridnet.generate_sample(sources, i+1)

        print('Epoch{:0>3d} '
              'train loss is {:.3f} '
              'train acc is {:.3f}% '
              'train conf acc is {:.3f}% '
              'train f1 score is {:.3f}% '.format(epoch+1,
                                                  Hybridnet.train_loss / (i + 1),
                                                  Hybridnet.train_acc / (i + 1) * 100,
                                                  Hybridnet.train_conf_acc / (i + 1) * 100,
                                                  Hybridnet.train_f1_score / (i + 1) * 100))

        torch.save({'state_dict': Hybridnet.model.state_dict(),
                    'loss': Hybridnet.train_loss / (i + 1),
                    'acc': Hybridnet.train_acc / (i + 1) * 100},
                   cfg.ckpt_path + '\\Epoch{:0>3d}_train_loss{:.3f}_train_acc{:.3f}.pth.tar'.format(
                       epoch + 1, Hybridnet.train_loss / (i + 1), Hybridnet.train_acc / (i + 1) * 100))

        Hybridnet.train_loss = 0
        Hybridnet.train_acc = 0
        Hybridnet.train_conf_acc = 0
        Hybridnet.train_f1_score = 0

        for i in range(data_gen.get_val_len()):
            sources, sg_sources, targets = next(validate_gen)
            Hybridnet.validate(sources, sg_sources, targets)

        print('Epoch{:0>3d} '
              'validate loss is {:.3f} '
              'validate acc is {:.3f}% '
              'validate conf acc is {:.3f}% '
              'validate f1 score is {:.3f}% '.format(epoch+1,
                                                     Hybridnet.val_loss / (i + 1),
                                                     Hybridnet.val_acc / (i + 1) * 100,
                                                     Hybridnet.val_conf_acc / (i + 1) * 100,
                                                     Hybridnet.val_f1_score / (i + 1) * 100))

        Hybridnet.val_loss = 0
        Hybridnet.val_acc = 0
        Hybridnet.val_conf_acc = 0
        Hybridnet.val_f1_score = 0
