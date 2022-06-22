#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/9/20 15:42
# @Author   : mayanmei
# @FileName : visualise_mask.py
# ============================================

import logging
import tensorflow as tf
import keras.backend as K
from keras.callbacks import Callback
import cv2
import os
import datetime
import numpy as np
from m_rcnn import common
from PIL import Image, ImageDraw, ImageFont
from m_rcnn.coco_dataset import CocoDataset
from utils.bbox_utils import BboxUtil
from utils.image_utils import ImageUtils
from utils.anchor_utils import AnchorUtils
from utils.mask_util import MaskUtil
from utils.misc_utils import MiscUtils
from m_rcnn.common import refine_detections_graph
import io
from config import cfg
logger = logging.getLogger(__name__)


class TBoardVisual(Callback):

    def __init__(self, tag, train_data, val_data, tboard_dir):
        super().__init__()
        self.tag = tag
        self.tboard_dir = tboard_dir
        self.debug_step = cfg.TRAIN.DEBUG_STEP
        self.detection_max_instances = cfg.TEST.DETECTION_MAX_INSTANCES
        self.class_names_path = cfg.COMMON.COCO_CLASS_NAMES_PATH
        self.class_names_list = self.read_class_name()
        self.font = ImageFont.truetype("data/font/simsun.ttc", 60)  # 设置字体
        self.mini_mask = cfg.TRAIN.USE_MINI_MASK
        self.max_gt_instances = cfg.TRAIN.MAX_GT_INSTANCES
        self.mean_pixel = np.array(cfg.COMMON.MEAN_PIXEL)
        self.train_data = train_data
        self.val_data = CocoDataset(cfg.TRAIN.COCO_VAL_ANN_PATH, cfg.TRAIN.COCO_VAL_IMAGE_PATH)
        #self.val_data = val_data
        self.bbox_util = BboxUtil()
        self.image_utils = ImageUtils()
        self.anchor_utils = AnchorUtils()
        self.mask_util = MaskUtil()
        self.misc_utils = MiscUtils()

    def read_class_name(self):
        with open(self.class_names_path, "r") as file:
            class_names_info = file.readlines()
            class_names_list = [class_names.strip() for class_names in class_names_info]
            print("class_names_list:",class_names_list)
            return class_names_list
        pass

    def on_batch_end(self, batch, logs=None):
        if batch==0: return

        data = self.val_data
        batch_size = len(data.image_info_list)
        print("加载验证集的数量：", batch_size)

        anchors = self.anchor_utils.generate_pyramid_anchors(image_shape=cfg.COMMON.IMAGE_SHAPE)

        windows_list = []
        molded_images_list = []
        image_metas_list = []
        for index, value in enumerate(data.image_info_list):
            image, image_meta, gt_class_ids, gt_boxes, gt_masks = self.bbox_util.load_image_gt(data, index, None, self.mini_mask)
            rpn_match, rpn_bbox = common.build_rpn_targets(anchors, gt_class_ids, gt_boxes)
            molded_images, image_metas, windows = self.image_utils.mode_input([image])
            print("-------------windows:", windows)

            if index == 0:
                batch_images = np.zeros((batch_size,) + image.shape, dtype=np.float32)
                batch_class_ids = np.zeros((batch_size, self.max_gt_instances), dtype=np.int32)
                batch_boxes = np.zeros((batch_size, self.max_gt_instances, 4), dtype=np.int32)
                batch_image_meta = np.zeros((batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_rpn_match = np.zeros([batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros([batch_size, cfg.TRAIN.ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
                batch_gt_masks = np.zeros((batch_size, gt_masks.shape[0], gt_masks.shape[1],
                                           self.max_gt_instances), dtype=gt_masks.dtype)

            batch_images[index] = self.image_utils.mold_image(image.astype(np.float32), self.mean_pixel)
            batch_class_ids[index, :gt_class_ids.shape[0]] = gt_class_ids
            batch_boxes[index, :gt_boxes.shape[0]] = gt_boxes
            batch_image_meta[index] = image_meta
            batch_rpn_match[index] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[index] = rpn_bbox
            batch_gt_masks[index, :, :, :gt_masks.shape[-1]] = gt_masks

            windows_list.append(windows)
            molded_images_list.append(molded_images)
            image_metas_list.append(image_metas)

            print("type(windows_list):{0},windows_list:{1}".format(np.array(windows_list).shape, windows_list))

        print("type(batch_images):{0},batch_class_ids:{1},batch_boxes:{2}".format(np.array(batch_images).shape,
                                                                           np.array(batch_class_ids).shape,
                                                                           np.array(batch_boxes).shape))

        # 模型预测的外接矩形框、类别、分数、mask区域

        #output_rois, mrcnn_class, mrcnn_bbox, mrcnn_mask = self.model.get_layer("proposal_targets").output

        mrcnn_class_logits = self.model.get_layer('mrcnn_class_logits').output
        mrcnn_class = self.model.get_layer('mrcnn_class').output
        mrcnn_bbox = self.model.get_layer('mrcnn_bbox').output
        mrcnn_mask = self.model.get_layer('mrcnn_mask').output
        rpn_rois = self.model.get_layer('ROI').output
        output_rois = self.model.get_layer('output_rois').output

        detections = self.model.get_layer('mrcnn_detection').output # 训练没有这个层，测试的时候才有

        functor = K.function(inputs=[self.model.input[0],
                                     self.model.input[1],
                                     self.model.input[2],
                                     self.model.input[3],
                                     self.model.input[4],
                                     self.model.input[5],
                                     self.model.input[6],
                                     K.learning_phase()],
                            outputs=[mrcnn_class_logits,mrcnn_class,mrcnn_bbox,mrcnn_mask,rpn_rois,output_rois])

        mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask, rpn_rois, output_rois = functor([batch_images,
                                                                                       batch_image_meta,
                                                                                       batch_rpn_match,
                                                                                       batch_rpn_bbox,
                                                                                       batch_class_ids,
                                                                                       batch_boxes,
                                                                                       batch_gt_masks,
                                                                                       True])

        print("type(mrcnn_class):{0},mrcnn_bbox:{1},mrcnn_mask:{2}".format(np.array(mrcnn_class).shape,np.array(mrcnn_bbox).shape,np.array(mrcnn_mask).shape))
        print("type(mrcnn_class_logits):{0},type(output_rois):{1}".format(np.array(mrcnn_class_logits).shape, np.array(output_rois).shape))
        # print("mrcnn_class_logits:", mrcnn_class_logits)
        print("mrcnn_class:", mrcnn_class)
        print("mrcnn_bbox:", mrcnn_bbox)
        #print("mrcnn_mask:", mrcnn_mask)
        print("rpn_rois:", rpn_rois)
        print("output_rois:", output_rois)

        print("type(batch_image_meta):", type(batch_image_meta))
        print("batch_image_meta:", batch_image_meta)
        print("type(image_metas_list):", type(image_metas_list))
        print("image_metas_list:", image_metas_list)

        # mrcnn_class =
        # mrcnn_bbox =
        detections = common.DetectionLayer(batch_size, name="mrcnn_detection")([rpn_rois,
                                                                                 mrcnn_class,
                                                                                 mrcnn_bbox,
                                                                                 np.array(batch_image_meta)])


        print("可视化detections:", detections)

        pred_class_ids = tf.argmax(mrcnn_class_logits, axis=2)
        print("pred_class_ids:", pred_class_ids)
        # pred_bbox = K.reshape(mrcnn_bbox, (-1, K.int_shape(mrcnn_bbox)[2], 4))
        # print("pred_bbox:", pred_bbox)

        writer = tf.summary.FileWriter(self.tboard_dir)

        # 遍历每一张图片在tboard上可视化
        for i, image_info in enumerate(data.image_info_list):
            print("image_info:",image_info)
            image = batch_images[i]

            original_image_shape = image.shape
            image_shape = (1024,1024,3)
            window = windows_list[i][0]
            print("windows_list:", windows_list)
            print("遍历window:", window)
            print(type(window))
            # todo:
            #window =

            window = self.bbox_util.norm_boxes(window, image_shape[:2])
            print("变换格式后的window:", window)
            print(type(window))
            wy1, wx1, wy2, wx2 = window
            shift = np.array([wy1, wx1, wy1, wx1])
            wh = wy2 - wy1  # window height
            ww = wx2 - wx1  # window width
            scale = np.array([wh, ww, wh, ww])
            # Convert boxes to normalized coordinates on the window
            boxes = mrcnn_bbox
            boxes = np.divide(boxes - shift, scale)
            boxes = self.bbox_util.denorm_boxes(boxes, original_image_shape[:2])
            print("boxes:", boxes)


            detections_batch = self.misc_utils.batch_slice([gt_boxes, mrcnn_class, mrcnn_bbox, window],
                                                           lambda x, y, w, z: refine_detections_graph(x, y, w, z),
                                                           2)
            detections = tf.reshape(detections_batch, [2, self.detection_max_instances, 6])
            print("detections:", detections)



            # origin_image_shape = image.shape
            # molded_image_shape = molded_images_list[i].shape
            # final_rois, final_class_ids, final_scores, final_masks = self.result_convert(batch_size,
            #                                                                              mrcnn_class_logits[i],
            #                                                                              mrcnn_mask[i],
            #                                                                              mrcnn_class[i],
            #                                                                              mrcnn_bbox[i],
            #                                                                              origin_image_shape,
            #                                                                              molded_image_shape,
            #                                                                              windows_list[i])
            #
            # class_names_list = self.class_names_list
            # pred_label_index = final_class_ids
            # pred_label = class_names_list[pred_label_index]
            # pred_box = final_rois
            # print("pred_label:", pred_label)
            # print("pred_box:", pred_box)

            gt_label = "invoice"
            pred_box = [50, 100, 500, 600]
            pred_label = "invoice"
            tf_img = self.make_image(image, gt_label, pred_label, pred_box)
            summary = tf.Summary(value=[tf.Summary.Value(tag="{}".format(self.tag),image=tf_img)])
            writer.add_summary(summary)
        writer.close()

        return


    def result_convert(self,batch_size,mrcnn_class_logits,mrcnn_class,mrcnn_bbox,mrcnn_mask,original_image_shape,molded_image_shape,window):
        n = batch_size
        print("n:", n)
        boxes = mrcnn_bbox[: n]
        print("boxes:", boxes)
        class_ids = mrcnn_class[: n].astype(np.int32)
        print("class_ids:", class_ids)
        scores = mrcnn_class_logits[: n]
        print("scores:", scores)

        window = self.bbox_util.norm_boxes(window, molded_image_shape[:2])
        print("window:", window)

        masks = mrcnn_mask[np.arange(n), :, :, class_ids]

        # window = self.bbox_util.norm_boxes(window, molded_image_shape[:2])
        # print("window:", window)
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])

        boxes = np.divide(boxes - shift, scale)
        boxes = self.bbox_util.denorm_boxes(boxes, original_image_shape[:2])

        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]

        # todo: 第一步，先过滤掉多预测出来的非目标的小框，这样的小框置信度比较低
        low_confidence = np.argwhere(scores < 0.73) # 卡证0.82

        if len(low_confidence) > 0:
            boxes = np.delete(boxes, low_confidence[0], axis=0)
            class_ids = np.delete(class_ids, low_confidence[0], axis=0)
            scores = np.delete(scores, low_confidence[0], axis=0)
            masks = np.delete(masks, low_confidence[0], axis=0)
        print("1-class_ids:", class_ids)

        # # todo: 第二步，过滤同一个区域中预测出多个类别中置信度低的
        boxes, class_ids, scores, masks = self.filter_box(boxes,class_ids,scores,masks)
        print("2-class_ids:", class_ids)
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)

        n = class_ids.shape[0]
        print("n:",n)

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(n):
            # Convert neural network mask to full size mask
            full_mask = self.mask_util.unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
            pass
        full_masks = np.stack(full_masks, axis=-1) if full_masks else np.empty(original_image_shape[:2] + (0,))

        return boxes, class_ids, scores, full_masks


    def filter_box(self, final_rois, final_class_ids, final_scores, final_masks):
        '''
        过滤同一个区域中预测出多个类别中置信度低的
        :param final_rois:
        :param final_class_ids:
        :param final_scores:
        :param final_masks:
        :return:
        '''

        length = len(final_rois)
        j_list = []
        for i in range(0, length):
            for j in range(i + 1, length):
                iou = self.calculate_IoU(final_rois[i], final_rois[j])
                if iou >= 0.5:
                    if final_scores[i] > final_scores[j]:
                        j_list.append(j)

        if len(j_list) > 0:
            for jj in j_list:
                final_rois = np.delete(final_rois, jj, axis=0)
                final_class_ids = np.delete(final_class_ids, jj, axis=0)
                final_scores = np.delete(final_scores, jj, axis=0)
                final_masks = np.delete(final_masks, jj, axis=0)

        return final_rois, final_class_ids, final_scores, final_masks


    @staticmethod
    def calculate_IoU(predicted_bound, ground_truth_bound):
        """
        computing the IoU of two boxes.
        Args:
            box: (xmin, ymin, xmax, ymax),通过左下和右上两个顶点坐标来确定矩形位置
        Return:
            IoU: IoU of box1 and box2.
        """
        pxmin, pymin, pxmax, pymax = predicted_bound
        gxmin, gymin, gxmax, gymax = ground_truth_bound
        parea = (pxmax - pxmin) * (pymax - pymin)  # 计算P的面积
        garea = (gxmax - gxmin) * (gymax - gymin)  # 计算G的面积

        # 求相交矩形的左下和右上顶点坐标(xmin, ymin, xmax, ymax)
        xmin = max(pxmin, gxmin)  # 得到左下顶点的横坐标
        ymin = max(pymin, gymin)  # 得到左下顶点的纵坐标
        xmax = min(pxmax, gxmax)  # 得到右上顶点的横坐标
        ymax = min(pymax, gymax)  # 得到右上顶点的纵坐标

        # 计算相交矩形的面积
        w = xmax - xmin
        h = ymax - ymin
        if w <= 0 or h <= 0:
            return 0

        area = w * h  # G∩P的面积
        # area = max(0, xmax - xmin) * max(0, ymax - ymin)  # 可以用一行代码算出来相交矩形的面积
        # 并集的面积 = 两个矩形面积 - 交集面积
        IoU = area / (parea + garea - area)

        return IoU


    def make_image(self, raw_image, label, pred, box, debug=False):

        height, width, channel = raw_image.shape
        image = Image.fromarray(cv2.cvtColor(np.uint8(raw_image), cv2.COLOR_BGR2RGB))
        #image = Image.fromarray(np.uint8(raw_image))
        draw = ImageDraw.Draw(image)

        draw.text((2,2), label, 'red', self.font)
        draw.text((width//2, 2), pred, 'blue', self.font)
        draw.rectangle(box, fill=None, outline='red', width=5)

        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.Summary.Image(height=height,
                                width=width,
                                colorspace=channel,
                                encoded_image_string=image_string)


# todo: 真实值和预测结果都是多个，要保证一一对应
def ids_accuracy(y_true, y_pred):
    '''
    计算类别的正确率
    :param y_pred:
    :return:
    '''
    max_idx_p = tf.argmax(y_pred, axis=2)
    max_idx_l = tf.argmax(y_true, axis=2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    _result = tf.map_fn(fn=lambda e: tf.reduce_all(e), elems=correct_pred, dtype=tf.bool)
    return tf.reduce_mean(tf.cast(_result, tf.float32))

# todo：用真实区域遍历预测区域
def calculate_iou(masks1, masks2):
    """
    :param masks1: [Height, Width, instances]
    :param masks2: [Height, Width, instances]
    :return: 两个 masks 的 IOU 重叠率
    """
    # 如果其中一个 masks 为空，则返回 空 结果
    mask_flag = masks1.shape[-1] == 0 or masks2.shape[-1] == 0
    if mask_flag:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
        pass

    # 将 masks 扁平化后并计算它们的面积
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps