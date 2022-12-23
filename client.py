# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import os
import time
import json
import numpy as np
import requests
from PIL import Image, ImageFont, ImageDraw
from _utils.utils import BBoxUtility
from _utils.anchors import Anchors
from configure import config as cfg

priors = Anchors(scales=cfg.scales,
                 ratios=cfg.ratios)(cfg.input_size)

class Client:
    def __init__(self,
                 root_url: str,
                 info_url: str,
                 model_url: str,
                 num_classes: int,
                 iou_thresh: float,
                 nms_thresh: float):
        """
        Used to request the docker service based on tensorflow serving
        """
        self.root_url = root_url
        self.info_url = info_url
        self.url = model_url % root_url
        self.box_utils = BBoxUtility(priors=priors,
                                     num_classes=num_classes,
                                     overlap_threshold=iou_thresh,
                                     nms_thresh=nms_thresh)

    def get_info(self):

        url = self.info_url % self.root_url
        response = requests.get(url)
        text = resp.text
        print("模型基本信息: ", text)

    def request(self, source):

        source = source.resize(reversed(cfg.input_size))
        source = np.array(source, dtype="float32")
        source = source/255.
        source = source.transpose([2, 0, 1])

        data = {
            "signature_name": "serving_default",
            "instances": [{"input": source.tolist()}]
        }
        ret = requests.post(self.url, json=data)
        if ret.status_code == 200:
            result = json.loads(ret.text)
            logits = result['predictions'][0]
            self.assign(source, logits)
        else:
            print("error")

    def assign(self, source, logits: dict):

        regressions, classifications, segmentations = logits["regressions"], \
                                                      logits["classifications"],\
                                                      logits["segmentations"]
        regressions = np.array(regressions)[np.newaxis]
        classifications = np.array(classifications)[np.newaxis]
        segmentations = np.array(segmentations)[np.newaxis]
        regressions = regressions.reshape(regressions.shape[0], -1, regressions.shape[-1])
        classifications = classifications.reshape(classifications.shape[0], -1, classifications.shape[-1])

        out_boxes, out_scores, out_classes = self.box_utils.detection_out([regressions,
                                                                           classifications])

        out_boxes = np.array(out_boxes).squeeze(axis=0)
        out_scores = np.array(out_scores).squeeze(axis=0)
        out_classes = np.array(out_classes).squeeze(axis=0)

        source = source.transpose([1, 2, 0])
        segmentation = segmentations[0].transpose([1, 2, 0])
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
        image.show()


if __name__ == '__main__':

    clinet = Client(root_url="http://127.0.0.1:8501",
                    info_url="%s/v1/models/hybridnet/metadata",
                    model_url="%s/v1/models/hybridnet:predict",
                    num_classes=cfg.class_names.__len__(),
                    iou_thresh=cfg.iou_thresh,
                    nms_thresh=cfg.nms_thresh)

    while True:
        file_path = input("Enter file path: ")
        try:
            source = Image.open(file_path)
            clinet.request(source)
        except Exception as e:
            print(repr(e))
