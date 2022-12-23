# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import itertools
import time

import numpy as np


class Anchors:

    def __init__(self,
                 scales: np.ndarray,
                 ratios: list,
                 anchor_scale=4.,
                 pyramid_levels=None):
        """
        :param pyramid_levels: 5 receptive fields by default
        """
        self.anchor_scale = anchor_scale

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels

        self.scales = scales
        self.ratios = ratios
        self.strides = [2 ** x for x in self.pyramid_levels]

    def __call__(self, image_shape, *args, **kwargs):
        """Generates multiscale anchor boxes.
        Args:
          image_size: integer number of input image size. The input image has the
            same dimension for width and height. The image_size should be divided by
            the largest feature stride 2^max_level.
          anchor_scale: float number representing the scale of size of the base
            anchor to the feature stride 2^level.
        Returns:
          anchor_boxes: a numpy array with shape [1, N, 4], which stacks anchors on all
            feature levels.
        """

        boxes_all = []
        for stride in self.strides:
            boxes_level = []
            for scale, ratio in itertools.product(self.scales, self.ratios):
                if image_shape[1] % stride != 0:
                    raise ValueError('input size must be divided by the stride.')
                base_anchor_size = self.anchor_scale * stride * scale
                anchor_size_x = base_anchor_size * ratio[0] / 2.0
                anchor_size_y = base_anchor_size * ratio[1] / 2.0

                x = np.arange(stride // 2, image_shape[1], stride)
                y = np.arange(stride // 2, image_shape[0], stride)
                ct_x, ct_y = np.meshgrid(x, y, indexing='xy')

                ct_yx = np.stack([ct_y, ct_x], axis=-1)
                ct_yx = np.tile(ct_yx, (2,)).astype('float')

                anchor_size = np.tile(np.array([anchor_size_y,
                                                anchor_size_x]), (2,)).astype('float')
                anchor_size[:2] = -anchor_size[:2]

                boxes = ct_yx + anchor_size
                boxes = boxes.reshape((-1, 4))

                boxes_level.append(boxes[:, np.newaxis])
            # concat anchors on the same level to shape Nxlx4
            boxes_level = np.concatenate(boxes_level, axis=1)
            boxes_all.append(boxes_level.reshape([-1, 4]))

        anchor_boxes = np.concatenate(boxes_all)

        anchor_boxes[:, 0::2] /= image_shape[1]
        anchor_boxes[:, 1::2] /= image_shape[0]

        return anchor_boxes
