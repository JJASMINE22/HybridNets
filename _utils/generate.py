import os
import json
import torch
import random
import numpy as np
from PIL import Image
from _utils.utils import get_random_data, BBoxUtility
from _utils import class_names, segmentation_class_names

class Generator:
    def __init__(self,
                 image_root: str,
                 anno_path: str,
                 input_size: tuple,
                 batch_size: int,
                 train_split: float,
                 priors: np.ndarray,
                 num_classes: int):
        """
        prior boxes tuning method based on retinaFace
        :param priors: the total prior boxes under each receptive field
        :param num_classes: number of detection categories
        """
        self.anno_path = anno_path
        self.image_root = image_root
        self.input_size = input_size
        self.batch_size = batch_size
        self.train_split = train_split
        self.split_train_val()
        self.box_util = BBoxUtility(priors=priors,
                                    num_classes=num_classes)

    def split_train_val(self):

        with open(self.anno_path, 'rb') as f:
            self.total_files_info = json.load(f)

        self.train_files_len = int(self.train_split * self.total_files_info.__len__())
        self.valid_files_len = self.total_files_info.__len__() - self.train_files_len

    def get_train_len(self):

        if not self.train_files_len % self.batch_size:
            return self.train_files_len // self.batch_size
        else:
            return self.train_files_len // self.batch_size + 1

    def get_val_len(self):

        if not self.valid_files_len % self.batch_size:
            return self.valid_files_len // self.batch_size
        else:
            return self.valid_files_len // self.batch_size + 1

    def generate(self, training=True):

        while True:
            random.shuffle(self.total_files_info)
            if training:
                files_info = self.total_files_info[:self.train_files_len]
            else:
                files_info = self.total_files_info[self.train_files_len:]

            sources, sg_sources, targets = [], [], []
            for i, file_info in enumerate(files_info):
                file_path = os.path.join(self.image_root, file_info['name'])

                boxes, sg_points, sg_names = [], [], []
                for annotation in file_info['labels']:
                    if annotation['category'] not in class_names + segmentation_class_names:
                        continue
                    try:
                        sg_points.append(annotation['poly2d'])
                        sg_names.append(annotation['category'])
                    except KeyError:

                        dt_points = annotation['box2d']
                        dt_points.update({"category": class_names.index(annotation['category'])})
                        boxes.append(list(map(lambda x: int(x), dt_points.values())))

                image, sg_image, boxes = get_random_data(file_path, np.array(boxes) if len(boxes) else boxes,
                                                         sg_points, sg_names, segmentation_class_names,
                                                         self.input_size)

                try:
                    one_hot_label = np.eye(class_names.__len__())[np.array(boxes)[:, -1].astype('int')]
                except IndexError:
                    pass

                if len(boxes):
                    boxes = np.concatenate([boxes[:, :4], one_hot_label], axis=-1)
                    del one_hot_label
                else:
                    pass

                assign_boxes = self.box_util.assign_boxes(boxes)

                sources.append(image)
                sg_sources.append(sg_image)
                targets.append(assign_boxes)

                if np.logical_or(np.equal(sources.__len__(), self.batch_size),
                                 np.equal(i + 1, files_info.__len__())):

                    sources_ = np.array(sources.copy()).transpose((0, 3, 1, 2))
                    sg_sources_ = np.eye(segmentation_class_names.__len__())[np.array(sg_sources.copy())]
                    targets_ = np.array(targets.copy())

                    sources.clear()
                    sg_sources.clear()
                    targets.clear()

                    yield sources_, sg_sources_, targets_
