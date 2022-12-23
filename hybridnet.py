# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import torch
import numpy as np
from torch import nn
from PIL import Image, ImageFont, ImageDraw
from net.networks import CreateModel
from custom.CustomLosses import ConfidenceLoss, BBOXL1Loss
from _utils.utils import BBoxUtility, calculate_score
from configure import config as cfg

class HybridNet:
    def __init__(self,
                 fpn_cells: int, num_layers: int,
                 num_anchors: int, num_classes: int,
                 seg_classes: int, num_features: int,
                 conv_channels: list, out_indices: tuple,
                 up_scale: tuple, backbone: str,
                 priors: np.ndarray, learning_rate: float,
                 weight_decay: float, iou_thresh: float,
                 nms_thresh: float, resume_train: bool,
                 ckpt_path: str):

        self.box_utils = BBoxUtility(priors=priors,
                                     num_classes=num_classes-1,
                                     overlap_threshold=iou_thresh,
                                     nms_thresh=nms_thresh)

        self.model = CreateModel(fpn_cells=fpn_cells,
                                 num_layers=num_layers,
                                 num_anchors=num_anchors,
                                 num_classes=num_classes,
                                 seg_classes=seg_classes,
                                 num_features=num_features,
                                 conv_channels=conv_channels,
                                 out_indices=out_indices,
                                 up_scale=up_scale,
                                 backbone=backbone)

        if cfg.device:
            self.model = self.model.to(cfg.device)

        if resume_train:
            try:
                ckpt = torch.load(ckpt_path)
                self.model.load_state_dict(ckpt['state_dict'])
                print("model successfully loaded, loss is {:.3f}".format(ckpt['loss']))
            except FileNotFoundError:
                raise ("please enter the right params path")

        self.conf_loss = ConfidenceLoss()
        self.bbox_loss = BBOXL1Loss()
        self.seg_loss = nn.BCELoss(reduction='mean')

        weights, bias = self.model.split_weights()
        self.optimizer = torch.optim.Adam(params=[{'params': weights, 'weight_decay': weight_decay},
                                                  {'params': bias}], lr=learning_rate)

        self.train_loss, self.val_loss = 0, 0
        self.train_acc, self.val_acc = 0, 0
        self.train_conf_acc, self.val_conf_acc = 0, 0
        self.train_f1_score, self.val_f1_score = 0, 0
        self.num_classes = num_classes - 1

    def train(self, sources, seg_sources, targets):

        sources = torch.tensor(sources).float()
        seg_sources = torch.tensor(seg_sources).float()
        targets = torch.tensor(targets).float()

        if cfg.device:
            sources = sources.to(cfg.device)
            seg_sources = seg_sources.to(cfg.device)
            targets = targets.to(cfg.device)

        self.optimizer.zero_grad()

        regressions, classifications, segmentations = self.model(sources)
        regressions = regressions.reshape(regressions.size(0), -1, regressions.size(-1))
        classifications = classifications.reshape(classifications.size(0), -1, classifications.size(-1))

        conf_loss = self.conf_loss(classifications, targets[..., 4:])
        bbox_loss = self.bbox_loss(regressions, targets[..., :5])
        seg_loss = self.seg_loss(segmentations.permute((0, 2, 3, 1)), seg_sources)

        loss = conf_loss + bbox_loss + seg_loss

        loss.backward()
        self.optimizer.step()

        self.train_loss += loss.data.item()

        prob_confs = torch.where(torch.ge(classifications[..., 0:1], .5),
                                 torch.ones_like(classifications[..., 0:1]),
                                 torch.zeros_like(classifications[..., 0:1]))

        total_num = torch.prod(torch.tensor(targets[..., 4:5].size())).data.item()
        object_num = (1 - targets[..., 4:5]).cpu().sum().data.item()

        correct_conf_num = torch.eq(targets[..., 4:5], prob_confs).float().detach().cpu().sum().data.item()

        self.train_conf_acc += correct_conf_num / total_num

        object_mask = (1 - targets[..., 4:5]).squeeze(dim=-1).bool()

        prob_class = classifications[..., 1:][object_mask].argmax(dim=-1)
        real_class = targets[..., 5:][object_mask].argmax(dim=-1)

        correct_class_num = torch.eq(real_class, prob_class).float().detach().cpu().sum().data.item()

        self.train_acc += correct_class_num / object_num

        self.train_f1_score += calculate_score(targets[..., 5:], classifications[..., 1:],
                                               object_mask, self.num_classes)

    def validate(self, sources, seg_sources, targets):

        sources = torch.tensor(sources).float()
        seg_sources = torch.tensor(seg_sources).float()
        targets = torch.tensor(targets).float()

        if cfg.device:
            sources = sources.to(cfg.device)
            seg_sources = seg_sources.to(cfg.device)
            targets = targets.to(cfg.device)

        regressions, classifications, segmentations = self.model(sources)
        regressions = regressions.reshape(regressions.size(0), -1, regressions.size(-1))
        classifications = classifications.reshape(classifications.size(0), -1, classifications.size(-1))

        conf_loss = self.conf_loss(classifications, targets[..., 4:])
        bbox_loss = self.bbox_loss(regressions, targets[..., :5])
        seg_loss = self.seg_loss(segmentations.permute((0, 2, 3, 1)), seg_sources)

        loss = conf_loss + bbox_loss + seg_loss

        self.val_loss += loss.data.item()

        prob_confs = torch.where(torch.ge(classifications[..., 0:1], .5),
                                 torch.ones_like(classifications[..., 0:1]),
                                 torch.zeros_like(classifications[..., 0:1]))

        total_num = torch.prod(torch.tensor(targets[..., 4:5].size())).data.item()
        object_num = (1 - targets[..., 4:5]).cpu().sum().data.item()

        correct_conf_num = torch.eq(targets[..., 4:5], prob_confs).float().detach().cpu().sum().data.item()

        self.val_conf_acc += correct_conf_num / total_num

        object_mask = (1 - targets[..., 4:5]).squeeze(dim=-1).bool()

        prob_class = classifications[..., 1:][object_mask].argmax(dim=-1)
        real_class = targets[..., 5:][object_mask].argmax(dim=-1)

        correct_class_num = torch.eq(real_class, prob_class).float().detach().cpu().sum().data.item()

        self.val_acc += correct_class_num / object_num

        self.val_f1_score += calculate_score(targets[..., 5:], classifications[..., 1:],
                                             object_mask, self.num_classes)

    def generate_sample(self, sources, batch):

        """
        Drawing and labeling
        """
        sources = torch.tensor(sources).float()
        if cfg.device:
            sources = sources.to(cfg.device)

        regressions, classifications, segmentations = self.model(sources)
        regressions = regressions.reshape(regressions.size(0), -1, regressions.size(-1))
        classifications = classifications.reshape(classifications.size(0), -1, classifications.size(-1))

        regressions = regressions.detach().cpu().numpy()
        classifications = classifications.detach().cpu().numpy()
        segmentations = segmentations.detach().cpu().numpy()

        index = np.random.choice(sources.size(0), 1)

        out_boxes, out_scores, out_classes = self.box_utils.detection_out([regressions[index],
                                                                           classifications[index]])

        out_boxes = np.array(out_boxes).squeeze(axis=0)
        out_scores = np.array(out_scores).squeeze(axis=0)
        out_classes = np.array(out_classes).squeeze(axis=0)

        source = sources[index[0]].cpu().numpy().transpose([1, 2, 0])
        segmentation = segmentations[index[0]].transpose([1, 2, 0])
        image = Image.fromarray(np.uint8(source * 255))

        if out_boxes.shape[0]:

            out_boxes = self.box_utils.correct_boxes(out_boxes, np.array(cfg.input_size),
                                                     np.array(cfg.input_size))

            out_boxes *= np.tile(np.array(cfg.input_size)[::-1], (2,))

            for coordinate, out_score, out_class in zip(out_boxes.astype('int'),
                                                        out_scores,
                                                        out_classes):

                left, top = coordinate[:2].tolist()
                right, bottom = coordinate[2:].tolist()

                font = ImageFont.truetype(font=cfg.font_path,
                                          size=np.floor(4e-2 * image.size[1] + 0.5).astype('int32'))

                label = '{:s}: {:.2f}'.format(cfg.class_names[int(out_class)], out_score)

                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
                label = label.encode('utf-8')

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                draw.rectangle(coordinate[:2].tolist() + coordinate[2:].tolist(),
                               outline=cfg.rect_color, width=int(2 * cfg.thickness))

                draw.text(text_origin, str(label, 'UTF-8'),
                          fill=cfg.font_color, font=font)
                del draw

        image = np.array(image)
        segmentation = segmentation.argmax(axis=-1)
        for i in range(cfg.segmentation_class_names.__len__()):
            if not i: continue
            image[np.equal(segmentation, i)] = cfg.segmentation_colors[i]

        image = Image.fromarray(image)

        # image.save(cfg.sample_path.format(batch), quality=95, subsampling=0)
        image.show()
