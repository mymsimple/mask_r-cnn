#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/05/18 14:42
# @Author   : WanDaoYi
# @FileName : mask_test.py
# ============================================

from datetime import datetime
import os
import cv2
import base64
import json
import numpy as np
import logging
from m_rcnn.mask_rcnn import MaskRCNN
from config import cfg

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logger = logging.getLogger("maskrcnn预测")


def init_logger():
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.DEBUG,
        handlers=[logging.StreamHandler()])


class MaskTest(object):

    def __init__(self):

        # 获取 类别 list
        # TODO !! 分类数据
        self.class_names_path = cfg.COMMON.COCO_CLASS_NAMES_PATH
        # self.class_names_path = cfg.COMMON.OUR_CLASS_NAMES_PATH
        self.class_names_list = self.read_class_name()

        # 测试图像的输入 和 输出 路径
        self.test_image_file_path = cfg.TEST.TEST_IMAGE_FILE_PATH
        self.output_image_path = cfg.TEST.OUTPUT_IMAGE_PATH
        self.debug_image_path = cfg.TEST.DEBUG_IMAGE_PATH
        self.prediction_path = cfg.TEST.PREDICTION_PATH

        # 加载网络模型
        self.mask_model = MaskRCNN(train_flag=False)
        # 加载权重模型
        self.mask_model.load_weights(cfg.TEST.COCO_MODEL_PATH, by_name=True)

        pass

    def read_class_name(self):
        with open(self.class_names_path, "r") as file:
            class_names_info = file.readlines()
            class_names_list = [class_names.strip() for class_names in class_names_info]
            print("class_names_list:", class_names_list)
            return class_names_list
        pass

    def get_images(self, data_path):
        '''
        find image files in test data path
        :return: list of files found
        '''
        files = []
        exts = ['jpg', 'png', 'jpeg', 'JPG']
        for parent, dirnames, filenames in os.walk(data_path):
            for filename in filenames:
                for ext in exts:
                    if filename.endswith(ext):
                        files.append(os.path.join(parent, filename))
                        break
        return files

    def do_test(self):
        """
            batch predict
        :param show_image_flag: show images or not
        :return:
        """
        test_image_name_list = self.get_images(self.test_image_file_path)

        labels = []
        for test_image_path in test_image_name_list:
            test_image_name = os.path.basename(test_image_path)
            image_info = cv2.imread(test_image_path)
            print("read img:", test_image_path)
            h, w, _ = image_info.shape
            print(image_info.shape)

            # Run detection
            results_info_list = self.mask_model.detect([image_info])
            print("results: {}".format(results_info_list))

            rois = results_info_list[0]['rois']
            class_ids = results_info_list[0]['class_ids']
            scores = results_info_list[0]['scores']
            masks = results_info_list[0]['masks']
            mask_cnt = masks.shape[-1]
            CLASS_NAME = self.class_names_list[1:]

            result = []
            for i in range(mask_cnt):
                confidence = np.float(scores[i])
                _rois = rois[i].tolist()
                box = self.cut_rectangle(_rois)
                box = np.int0(box)
                # 画线
                cv2.polylines(image_info, [box], True, (255, 0, 0), 3)
                # save images
                output_image_path = os.path.join(self.output_image_path, test_image_name)
                cv2.imwrite(output_image_path, image_info)

                class_points = {
                    "label": CLASS_NAME[class_ids[i] - 1],
                    "scores": confidence,
                    "rois": _rois,
                    "points": box.tolist(),
                    "group_id": " ",
                    "shape_type": "polygon",
                    "flags": {}
                }
                result.append(class_points)
                prediction = {"version": "3.16.7",
                              "flags": {},
                              'shapes': result,
                              "imagePath": test_image_name,
                              # "imageData": self.nparray2base64(image_info),
                              "imageHeight": h,
                              "imageWidth": w
                              }

            # image_path = os.path.join(self.debug_image_path + test_image_name)
            # cv2.imwrite(image_path, image_info)

            prediction_json_path = os.path.join(self.prediction_path + test_image_name[:-4] + ".json")
            with open(prediction_json_path, "w", encoding='utf-8') as g:
                json.dump(prediction, g, indent=2, sort_keys=True, ensure_ascii=False)
        pass

    @staticmethod
    def nparray2base64(img_data):
        """
            nparray格式的图片转为base64（cv2直接读出来的就是）
        :param img_data:
        :return:
        """
        _, d = cv2.imencode('.jpg', img_data)
        return str(base64.b64encode(d), 'utf-8')


    def cut_rectangle(self, box):
        '''
        抠出外接矩形
        :return:
        '''

        # mask = results_info_list[0]['masks']
        ymin = box[0]
        xmin = box[1]
        ymax = box[2]
        xmax = box[3]
        # w = xmax - xmin
        # h = ymax - ymin
        # cut_img = image_info[ymin:ymin+h, xmin:xmin+w]
        # cut_img_path = "data/idcard/test/cut/"
        # cv2.imwrite(os.path.join(cut_img_path + test_image_name), cut_img)

        point1 = [xmin, ymin]
        point2 = [xmax, ymax]
        # 将2点变为4点
        point_1 = [point1[0], point1[1]]
        point_2 = [point2[0], point1[1]]
        point_3 = [point2[0], point2[1]]
        point_4 = [point1[0], point2[1]]

        # 将新的点坐标填入points
        points = [point_1, point_2, point_3, point_4]
        return points
        pass



class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


if __name__ == "__main__":
    init_logger()

    # 代码开始时间
    start_time = datetime.now()
    print("开始时间: {}".format(start_time))

    demo = MaskTest()
    demo.do_test()

    # 代码结束时间
    end_time = datetime.now()
    print("结束时间: {}, 测试模型耗时: {}".format(end_time, end_time - start_time))
