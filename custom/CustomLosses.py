# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import torch
from torch import nn
from torch.nn import functional as F

class ConfidenceLoss(nn.Module):
    def __init__(self,
                 neg_pos_ratio: int = 4,
                 neg_for_hard: int = 100):
        super(ConfidenceLoss, self).__init__()
        # self.loss_func = nn.BCEWithLogitsLoss(reduction='none')
        self.neg_pos_ratio = neg_pos_ratio
        self.neg_for_hard = neg_for_hard

    def forward(self, y_pred, y_true):

        def softmax_loss(y_pred, y_true):
            """
            This error is only applicable to the output activated by softmax
            :param y_true: one hot label
            :return: softmax loss
            """
            y_pred = torch.maximum(y_pred, torch.tensor(1e-7))
            softmax_loss = -torch.sum(y_true * torch.log(y_pred), dim=-1)

            return softmax_loss

        # -------------------------------#
        #   取出先验框的数量
        # -------------------------------#
        num_boxes = y_true.size(1)
        # --------------------------------------------- #
        #   softmax loss
        # --------------------------------------------- #
        cls_loss = softmax_loss(y_pred, y_true)
        # cls_loss = self.loss_func(y_pred, y_true).sum(dim=-1)
        # --------------------------------------------- #
        #   每一张图的正样本的个数
        #   (batch_size,)
        # --------------------------------------------- #
        num_pos = torch.sum(1 - y_true[..., 0], dim=-1)

        pos_conf_loss = torch.sum(cls_loss * (1 - y_true[..., 0]), dim=1)
        # --------------------------------------------- #
        #   多数情况下大部分候选框都不包含检测物, 导致负样本误差极大, 极易造成神经元死亡
        #   每一张图的负样本的个数
        #   batch_size
        # --------------------------------------------- #
        num_neg = torch.minimum(self.neg_pos_ratio * num_pos, num_boxes - num_pos)
        # 找到了哪些值是大于0的
        pos_num_neg_mask = torch.greater(num_neg, 0)
        # --------------------------------------------- #
        #   如果所有的图，正样本的数量均为0
        #   那么则默认选取100个先验框作为负样本
        # --------------------------------------------- #
        has_min = torch.any(pos_num_neg_mask).float()
        num_neg = torch.cat([num_neg,
                             torch.tensor([(1 - has_min) * self.neg_for_hard]).to(torch.device('cuda'))], dim=0)

        num_neg_batch = torch.sum(num_neg[torch.greater(num_neg, 0)])
        num_neg_batch = num_neg_batch.int()

        # --------------------------------------------- #
        #   batch_size, k
        #   把不是背景的概率求和，求和后的概率越大, 代表越难分类
        #   使用softmax loss判断亦可
        # --------------------------------------------- #
        max_confs = torch.sum(y_pred[..., 1:], axis=-1)
        # --------------------------------------------------- #
        #   只有没有包含物体的先验框才得到保留
        #   我们在整个batch里面选取最难分类的num_neg_batch个
        #   先验框作为负样本。
        # --------------------------------------------------- #
        max_confs = torch.reshape(max_confs * y_true[..., 0], [-1])
        indices = torch.topk(max_confs, k=num_neg_batch).indices

        # 根据索引, 取出num_neg_batch个误差最大的负样本
        neg_conf_loss = torch.reshape(cls_loss, [-1])[indices]

        # 进行归一化
        num_pos = torch.where(torch.not_equal(num_pos, 0), num_pos, torch.ones_like(num_pos))
        total_loss = torch.sum(pos_conf_loss) + torch.sum(neg_conf_loss)
        total_loss /= torch.sum(num_pos)

        return total_loss

class BBOXL1Loss(nn.Module):
    def __init__(self,
                 sigma: int = 1,
                 weights: int = 1):
        super(BBOXL1Loss, self).__init__()
        self.sigma_squared = sigma ** 2
        self.weights = weights

    def forward(self, y_pred, y_true):

        regression_logits = y_pred[..., :4]  # y_pred shape: [bs, len(anchors), 4]
        regression_targets = y_true[..., :4]  # y_true shape: [bs, len(anchors), 5]
        anchor_state = 1 - y_true[..., 4]  # y_true索引4包含目标状态

        # ------------------------------------#
        #   取出作为正样本的先验框
        # ------------------------------------#
        # indices = torch.where(torch.eq(anchor_state, 1))
        # indices = torch.stack(indices, dim=-1)
        bool_mask = torch.eq(anchor_state, 1)
        regression_logits = regression_logits[bool_mask]
        regression_targets = regression_targets[bool_mask]

        # ------------------------------------#
        #   计算 smooth L1 loss
        # ------------------------------------#
        regression_diff = regression_logits - regression_targets
        regression_diff = torch.abs(regression_diff)
        regression_loss = torch.where(
            torch.less(regression_diff, 1.0 / self.sigma_squared),
            0.5 * self.sigma_squared * torch.pow(regression_diff, 2),
            regression_diff - 0.5 / self.sigma_squared)

        normalizer = torch.maximum(torch.ones(size=()), torch.tensor(regression_diff.size(0)))
        normalizer = normalizer.float()
        loss = torch.sum(regression_loss) / normalizer

        return loss * self.weights

