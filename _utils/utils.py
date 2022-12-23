import torch
import itertools
import numpy as np
from torch import nn
from PIL import Image, ImageDraw

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    return class_names

def get_segmentation_classes(segmentation_classes_path):
    '''loads the segmentation classes'''
    with open(segmentation_classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    return class_names


class BBoxUtility(object):
    def __init__(self,
                 priors: np.ndarray,
                 num_classes: int,
                 overlap_threshold=0.5,
                 nms_thresh=0.5):
        self.priors = priors
        self.num_classes = num_classes
        self.num_priors = 0 if priors is None else len(priors)
        self.overlap_threshold = overlap_threshold
        self._nms_thresh = nms_thresh

    def iou(self, box, priors):
        # 计算出每个真实框与所有的先验框的iou
        # 判断真实框与先验框的重合情况
        inter_upleft = np.maximum(priors[:, :2], box[:2])
        inter_botright = np.minimum(priors[:, 2:4], box[2:])

        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # 真实框的面积
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        # 先验框的面积
        area_gt = (priors[:, 2] - priors[:, 0]) * (priors[:, 3] - priors[:, 1])
        # 计算iou
        union = area_true + area_gt - inter

        iou = inter / np.maximum(union, 1e-6)

        return iou

    def encode_box(self, box, return_iou=True):
        iou = self.iou(box[:4], self.priors)
        encoded_box = np.zeros((self.num_priors, 4 + return_iou))
        # ---------------------------------------------------#
        #   找到每一个真实框，重合程度较高的先验框
        # ---------------------------------------------------#
        assign_mask = iou > self.overlap_threshold
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        if return_iou:
            encoded_box[:, 4][assign_mask] = iou[assign_mask]

        # 找到对应的先验框
        assigned_priors = self.priors[assign_mask]

        # ----------------------------------------------------#
        #   逆向编码，将真实框转化为Retinaface预测结果的格式
        #   先计算真实框的中心与长宽
        # ----------------------------------------------------#
        box_center = 0.5 * (box[:2] + box[2:4])
        box_wh = box[2:4] - box[:2]
        # ---------------------------------------------#
        #   再计算重合度较高的先验框的中心与长宽
        # ---------------------------------------------#
        assigned_priors_center = 0.5 * (assigned_priors[:, :2] +
                                        assigned_priors[:, 2:4])
        assigned_priors_wh = (assigned_priors[:, 2:4] -
                              assigned_priors[:, :2])

        # ------------------------------------------------#
        #   逆向求取应该有的预测结果
        # ------------------------------------------------#
        encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
        encoded_box[:, :2][assign_mask] /= assigned_priors_wh
        encoded_box[:, :2][assign_mask] /= 0.1

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_priors_wh)
        encoded_box[:, 2:4][assign_mask] /= 0.2

        return encoded_box.ravel()

    def assign_boxes(self, boxes):

        assignment = np.zeros((self.num_priors, 4 + 1 + self.num_classes))
        # ---------------------------------#
        #   索引4的位置代表是否为背景
        # ---------------------------------#
        assignment[:, 4] = 1
        if len(boxes) == 0:
            return assignment

        # -------------------------------------#
        #   每一个真实框的编码后的值，和iou
        #   encoded_boxes   n, num_priors, 5 + num_classes
        # -------------------------------------#
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes)
        encoded_boxes = encoded_boxes.reshape(-1, self.num_priors, 5)

        # -----------------------------------------------------#
        #   取重合程度最大的先验框，并且获取这个先验框的index
        #   num_priors,
        # -----------------------------------------------------#
        best_iou = encoded_boxes[:, :, 4].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, 4].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]

        assign_num = len(best_iou_idx)
        # 依赖于numpy的双索引, 得到唯一先验框标签
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]

        assignment[:, 4][best_iou_mask] = 0
        assignment[:, 5:][best_iou_mask] = boxes[best_iou_idx, 4:]

        return assignment

    @classmethod
    def correct_boxes(cls, boxes, input_shape: np.ndarray, image_shape: np.ndarray):
        """
        When the input is added with letterboxes, restore the coordinates when outputting.
        Because the input size of each sample in a batch is different, only process single sample
        """
        new_shape = image_shape * np.min(input_shape / image_shape)

        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape

        scale_for_boxs = np.tile(scale[::-1], (2,)) # [scale[1], scale[0], scale[1], scale[0]]

        offset_for_boxs = np.tile(offset[::-1], (2,)) # [offset[1], offset[0], offset[1], offset[0]]

        boxes = (boxes - np.array(offset_for_boxs)) * np.array(scale_for_boxs)

        return boxes

    def decode_boxes(self, mbox_loc):
        # 获得先验框的宽与高
        prior_wh = self.priors[:, 2:4] - self.priors[:, :2]

        # 获得先验框的中心点
        prior_center = (self.priors[:, :2] + self.priors[:, 2:4]) * 0.5

        # 真实框距离先验框中心的xy轴偏移情况
        decode_bbox_center = mbox_loc[:, :2] * prior_wh * 0.1
        decode_bbox_center += prior_center

        # 真实框的宽与高的求取
        decode_bbox_wh = np.exp(mbox_loc[:, 2:4] * 0.2)
        decode_bbox_wh *= prior_wh

        # 获取真实框的左上角与右下角
        decode_bbox_xy_min = decode_bbox_center - 0.5 * decode_bbox_wh
        decode_bbox_xy_max = decode_bbox_center + 0.5 * decode_bbox_wh

        # 真实框的左上角与右下角进行堆叠
        decode_bbox = np.concatenate([decode_bbox_xy_min, decode_bbox_xy_max], axis=-1)
        # 防止超出0与1
        decode_bbox = np.clip(decode_bbox, 0., 1.)

        return decode_bbox

    def detection_out(self, predictions, conf_thresh=0.5, prob_thresh=0.4):
        # ---------------------------------------------------#
        #   预测结果分为两部分，0为回归预测结果
        #   1为分类预测结果
        # ---------------------------------------------------#
        mbox_loc = predictions[0]
        mbox_conf = predictions[1]

        total_boxes, total_scores, total_classes = [], [], []
        for i in range(mbox_loc.__len__()):
            # ------------------------------------------------#
            #   解码过程
            # ------------------------------------------------#
            decode_bbox = self.decode_boxes(mbox_loc[i])

            """
            class_conf = 1 - mbox_conf[i][:, 0]

            conf_mask = (class_conf >= conf_thresh)

            class_prob = np.max(mbox_conf[i][:, 1:], axis=1)[conf_mask]
            class_pred = np.argmax(mbox_conf[i][:, 1:], axis=1)[conf_mask]
            decode_bbox = decode_bbox[conf_mask]

            prob_mask = (class_prob >= prob_thresh)

            detections = np.concatenate([decode_bbox[prob_mask], 
                                         class_prob[prob_mask],
                                         class_pred[prob_mask]],
                                        axis=1)
            """
            # 置信度矩阵的索引0位置代表是否为背景, 因此从索引1开始取
            total_class_conf = mbox_conf[i][:, 1:]

            class_conf = np.max(total_class_conf, axis=1)[..., np.newaxis]
            class_pred = np.argmax(total_class_conf, axis=1)[..., np.newaxis]
            # --------------------------------#
            #   判断置信度是否大于门限要求
            # --------------------------------#
            conf_mask = (class_conf >= conf_thresh)[:, 0]

            # --------------------------------#
            #   将预测结果进行堆叠
            # --------------------------------#
            detections = np.concatenate([decode_bbox[conf_mask],
                                         class_conf[conf_mask],
                                         class_pred[conf_mask]],
                                        axis=1)
            unique_class = np.unique(detections[:, -1])

            best_box, best_score, best_class = [], [], []
            if not unique_class.__len__():
                total_boxes.append(best_box)
                total_scores.append(best_score)
                total_classes.append(best_class)
                continue
            # ---------------------------------------------------------------#
            #   对种类进行循环,
            #   非极大抑制的作用是筛选出一定区域内属于同一种类得分最大的框,
            #   对种类进行循环可以帮助我们对每一个类分别进行非极大抑制
            # ---------------------------------------------------------------#
            for cls in unique_class:
                cls_mask = detections[:, -1] == cls
                detection = detections[cls_mask]
                scores = detection[:, 4]
                points = detection[..., :4]
                # ------------------------------------------#
                #   根据得分对该种类进行从大到小排序
                # ------------------------------------------#
                arg_sort = np.argsort(scores)[::-1]
                points = points[arg_sort]
                scores = scores[arg_sort]
                while np.shape(points)[0] > 0:
                    # -------------------------------------------------------------------------------------#
                    #   每次取出得分最大的框，计算其与其它所有预测框的重合程度，重合程度过大的则剔除。
                    # -------------------------------------------------------------------------------------#
                    best_box.append(points[0])
                    best_score.append(scores[0])
                    best_class.append(cls)
                    if len(points) == 1:
                        break
                    ious = self.iou(best_box[-1], points[1:])
                    points = points[1:][ious < self._nms_thresh]
                    scores = scores[1:][ious < self._nms_thresh]
            total_boxes.append(best_box)
            total_scores.append(best_score)
            total_classes.append(best_class)

        return total_boxes, total_scores, total_classes


def get_random_data(image_path, boxes, sg_points,
                    sg_names, seg_class_names, input_shape):
    '''random preprocessing for real-time data augmentation'''
    image = Image.open(image_path)
    iw, ih = image.size
    h, w = input_shape

    # resize image
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (w - nw) // 2
    dy = (h - nh) // 2

    sg_image = Image.new("L", image.size)
    draw = ImageDraw.Draw(sg_image)
    for name, points in zip(sg_names, sg_points):
        for point in points:
            closed = point['closed']
            point = np.array(point['vertices'], 'float')
            point = point.tolist()
            point = [tuple(pt) for pt in point]
            # 分别*10与*15的目的在于避免图像resize时, 闭环与非闭环分割区域的边界效应
            # uint8下值为1与2中间的像素将会被置为1, 导致图像分割时误判
            if closed:
                draw.polygon(point, fill=seg_class_names.index(name) * 15)
            else:
                for i in range(point.__len__() - 1):
                    draw.line([point[i], point[i+1]], seg_class_names.index(name) * 10, 13)

    del draw

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = np.array(new_image, np.float32) / 255.
    image = np.clip(image, 0., 1.)

    sg_image = sg_image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('L', (w, h))
    new_image.paste(sg_image, (dx, dy))
    new_image = np.array(new_image, "uint8")
    # 避免边界效应
    sg_image = np.zeros_like(new_image)
    bool_mask = np.logical_or(np.equal(new_image, 10),
                              np.equal(new_image, 30))
    sg_image[bool_mask] = new_image[bool_mask]
    sg_image[np.equal(sg_image, 30)] = 2
    sg_image[np.equal(sg_image, 10)] = 1

    # correct boxes
    if len(boxes) > 0:
        np.random.shuffle(boxes)
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * nw / iw + dx
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * nh / ih + dy
        boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
        boxes[:, 2][boxes[:, 2] > w] = w
        boxes[:, 3][boxes[:, 3] > h] = h
        boxes_w = boxes[:, 2] - boxes[:, 0]
        boxes_h = boxes[:, 3] - boxes[:, 1]
        boxes = boxes[np.logical_and(boxes_w > 1, boxes_h > 1)]  # discard invalid boxes

        boxes = boxes.astype('float')
        boxes[:, 0] = boxes[:, 0] / w
        boxes[:, 1] = boxes[:, 1] / h
        boxes[:, 2] = boxes[:, 2] / w
        boxes[:, 3] = boxes[:, 3] / h

    return image, sg_image, boxes


def calculate_score(y_true, y_pred, object_mask, depth: int):

    scores = []
    if object_mask.any():
        y_true = y_true[object_mask]
        y_pred = y_pred[object_mask]
        true_class = y_true.argmax(dim=-1)
        pred_class = y_pred.argmax(dim=-1)
        # 遍历检测物种类
        for i in range(depth):
            precision_num = torch.eq(pred_class, i).float().sum()
            if precision_num:

                true_bool_mask = torch.eq(true_class, i)
                pred_bool_mask = torch.eq(pred_class, i)

                bool_mask = torch.stack([true_bool_mask, pred_bool_mask], dim=-1).all(dim=-1)
                true_positive_num = bool_mask.float().sum()

            else:
                continue

            false_negative_num = true_bool_mask.float().sum() - true_positive_num
            recall_num = true_positive_num + false_negative_num

            if recall_num:

                precision = true_positive_num / precision_num
                recall = true_positive_num / recall_num

            else:
                continue

            score = 2 * (precision * recall) / (precision + recall)
            scores.append(score)

    f1_score = torch.square(torch.mean(torch.tensor(scores)))

    return torch.zeros(size=()) if torch.isnan(f1_score) else f1_score
