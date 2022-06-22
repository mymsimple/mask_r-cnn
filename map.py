# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
# 参考：https://www.jianshu.com/p/d7a06a720a2b
# https://github.com/facebookresearch/Detectron/blob/05d04d3a024f0991339de45872d02f2f50669b3d/lib/datasets/voc_eval.py#L54

import xml.etree.ElementTree as ET #读取xml文件
import os
import cv2
import json
import pickle
import numpy as np

'''
    计算ap、mAP
'''
# todo:解析json文件，将GT框信息放入一个列表
def parse_json(filename, bbox_path):
    objects = []
    bbox_json_path = os.path.join(bbox_path + filename + ".json")
    with open(bbox_json_path, "r") as f:
        json_data = json.load(f)
        for category in json_data['shapes']:
            obj_struct = {}
            obj_struct['name'] = category['label']
            obj_struct['bbox'] = __convetr(category['points'])
            obj_struct['difficult'] = 0
            objects.append(obj_struct)

    return objects

## 四点坐标转[xmin,ymin,xmax,ymax]
def __convetr(points):
    x = []
    y = []
    for point in points:
        x0 = point[0]
        y0 = point[1]
        x.append(x0)
        y.append(y0)
    xmin = min(x)
    xmax = max(x)
    ymin = min(y)
    ymax = max(y)
    return [ymin,xmin,ymax,xmax]


# 单个计算AP的函数，输入参数为精确率和召回率，原理见上面
def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    # 如果使用2017年的计算AP的方式(插值的方式)
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
       # 使用2010年后的计算AP值的方式
        # 这里是新增一个(0,0)，方便计算
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

# 主函数
def voc_eval(detpath, imagesetfile, classname, cachedir, ovthresh=0.9, use_07_metric=False):
    """
    rec, prec, ap = voc_eval(detpath,annopath,imagesetfile,classname,[ovthresh],[use_07_metric])
    detpath: 产生的txt文件，里面是一张图片的各个检测框结果。
    imagesetfile: 一个txt文件，里面是每个图片的地址，每行一个地址。
    classname: 种类的名字，即类别。
    cachedir: 缓存标注的目录。
    [ovthresh]: IOU阈值，默认为0.5，即mAP50。
    [use_07_metric]: 是否使用2007的计算AP的方法，默认为Fasle
    """

    # 首先加载Ground Truth标注信息
    # 读取文本里的所有图片路径
    with open(imagesetfile, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 获取文件名，strip用来去除头尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
    imagenames = [x.strip() for x in lines]


    recs = {}  # 存储Ground Truth标注信息
    for i, imagename in enumerate(imagenames):
        if imagename == ".DS_Store": continue
        else:
            name, ext = os.path.splitext(imagename)
            recs[imagename] = parse_json(name, cachedir)
    # # save file ，后面未用到
    # print('Saving cached annotations to {:s}'.format(cachefile))
    #
    # cachefile = os.path.join(cachedir, 'annots.pkl')
    # with open(cachefile, 'wb') as f:
    #     #写入pickle文件里面。写入的是一个字典，左侧为xml文件名，右侧为文件里面个各个参数。
    #     pickle.dump(recs, f)

    # 对每张图片的xml获取函数指定类的bbox等
    class_recs = {} # 保存的是某一类别的 Ground Truth的数据
    npos = 0
    for imagename in imagenames:
        # 获取Ground Truth每个文件中某种类别的物体
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        #  different基本都为0/False
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult) # 自增，~difficult取反,统计样本个数
        # # 记录Ground Truth的内容
        if len(det) > 0:
            class_recs[imagename] = {'bbox': bbox,
                                     'difficult': difficult,
                                     'det': det}

    # read dets 读取某类别预测输出
    detpath = os.path.join(detpath + classname + '.txt')
    #detpath = os.path.join("data/invoice/test/ids/" + classname + '.txt')
    with open(detpath, 'rb') as f:
        lines = f.readlines()
    splitlines = [x.decode().strip().split('  ') for x in lines]
    image_ids = [x[0] for x in splitlines] # 图片ID
    confidence = np.array([float(x[1]) for x in splitlines]) # IOU值
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]) # bounding box数值

    # 对confidence的index根据值大小进行降序排列
    sorted_ind = np.argsort(-confidence) # 降序，返回索引
    sorted_scores = np.sort(-confidence)  # 降序排列，返回置信度
    BB = BB[sorted_ind, :]  # 重排bbox，由大概率到小概率
    image_ids = [image_ids[x] for x in sorted_ind] # 图片重排，由大概率到小概率

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        if image_ids[d] not in class_recs.keys():continue
        else:
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # 计算IOU
                iymin = np.maximum(BBGT[:, 0], bb[0])
                ixmin = np.maximum(BBGT[:, 1], bb[1])
                iymax = np.minimum(BBGT[:, 2], bb[2])
                ixmax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih  # 交集
                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni  # 交并比
                ovmax = np.max(overlaps) # 最大重合率
                jmax = np.argmax(overlaps) # 最大重合率对应的gt

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    print('on {}, the ap is {}, recall is {}, precision is {}'.format(classname, ap, rec[-1], prec[-1]))
    return rec, prec, ap


# json---->txt
# name  confidence  bbox
def convert(json_path,prediction_txt_path):
    files = os.listdir(json_path)
    bboxes = []
    id_1_bboxes = []
    id_2_bboxes = []
    bank_1_bboxes = []
    bank_2_bboxes = []
    xsz_1_bboxes = []
    xsz_2_bboxes = []
    xsz_3_bboxes = []
    jsz_1_bboxes = []
    jsz_2_bboxes = []

    for file in files:
        print("file:", file)
        name,ext = os.path.splitext(file)
        bbox_json_path = os.path.join(json_path + file)
        with open(bbox_json_path, "r") as f:
            json_data = json.load(f)
            for category in json_data['shapes']:
                score = category['scores']
                ids = category['label']
                box = category['rois']

                result = name + ".jpg" + '  ' + str(score) + '  ' + str(box[0]).strip('[').strip(']') + \
                                 '  ' + str(box[1]).strip('[').strip(']') + '  ' + str(box[2]).strip('[').strip(']') + \
                                 '  ' + str(box[3]).strip('[').strip(']')
                bboxes.append(result)

                if ids == "id-1":
                    id_1_result = name + ".jpg" + '  ' + str(score) + '  ' + str(box[0]).strip('[').strip(']') + \
                             '  ' + str(box[1]).strip('[').strip(']') + '  ' + str(box[2]).strip('[').strip(']') + \
                             '  ' + str(box[3]).strip('[').strip(']')
                    id_1_bboxes.append(id_1_result)

                if ids == "id-2":
                    id_2_result = name + ".jpg" + '  ' + str(score) + '  ' + str(box[0]).strip('[').strip(']') + \
                             '  ' + str(box[1]).strip('[').strip(']') + '  ' + str(box[2]).strip('[').strip(']') + \
                             '  ' + str(box[3]).strip('[').strip(']')
                    id_2_bboxes.append(id_2_result)

                if ids == "bank-1":
                    bank_1_result = name + ".jpg" + '  ' + str(score) + '  ' + str(box[0]).strip('[').strip(']') + \
                             '  ' + str(box[1]).strip('[').strip(']') + '  ' + str(box[2]).strip('[').strip(']') + \
                             '  ' + str(box[3]).strip('[').strip(']')
                    bank_1_bboxes.append(bank_1_result)

                if ids == "bank-2":
                    bank_2_result = name + ".jpg" + '  ' + str(score) + '  ' + str(box[0]).strip('[').strip(']') + \
                             '  ' + str(box[1]).strip('[').strip(']') + '  ' + str(box[2]).strip('[').strip(']') + \
                             '  ' + str(box[3]).strip('[').strip(']')
                    bank_2_bboxes.append(bank_2_result)

                if ids == "xsz-1":
                    xsz_1_result = name + ".jpg" + '  ' + str(score) + '  ' + str(box[0]).strip('[').strip(']') + \
                             '  ' + str(box[1]).strip('[').strip(']') + '  ' + str(box[2]).strip('[').strip(']') + \
                             '  ' + str(box[3]).strip('[').strip(']')
                    xsz_1_bboxes.append(xsz_1_result)

                if ids == "xsz-2":
                    xsz_2_result = name + ".jpg" + '  ' + str(score) + '  ' + str(box[0]).strip('[').strip(']') + \
                             '  ' + str(box[1]).strip('[').strip(']') + '  ' + str(box[2]).strip('[').strip(']') + \
                             '  ' + str(box[3]).strip('[').strip(']')
                    xsz_2_bboxes.append(xsz_2_result)

                if ids == "xsz-3":
                    xsz_3_result = name + ".jpg" + '  ' + str(score) + '  ' + str(box[0]).strip('[').strip(']') + \
                             '  ' + str(box[1]).strip('[').strip(']') + '  ' + str(box[2]).strip('[').strip(']') + \
                             '  ' + str(box[3]).strip('[').strip(']')
                    xsz_3_bboxes.append(xsz_3_result)

                if ids == "jsz-1":
                    jsz_1_result = name + ".jpg" + '  ' + str(score) + '  ' + str(box[0]).strip('[').strip(']') + \
                             '  ' + str(box[1]).strip('[').strip(']') + '  ' + str(box[2]).strip('[').strip(']') + \
                             '  ' + str(box[3]).strip('[').strip(']')
                    jsz_1_bboxes.append(jsz_1_result)

                if ids == "jsz-2":
                    jsz_2_result = name + ".jpg" + '  ' + str(score) + '  ' + str(box[0]).strip('[').strip(']') + \
                             '  ' + str(box[1]).strip('[').strip(']') + '  ' + str(box[2]).strip('[').strip(']') + \
                             '  ' + str(box[3]).strip('[').strip(']')
                    jsz_2_bboxes.append(jsz_2_result)


    # with open(prediction_txt_path, "w", encoding='utf-8') as file:
    #     for box in bboxes:
    #         file.write(str(box) + "\n")

    with open(os.path.join(prediction_txt_path,"id-1" + ".txt"), "w", encoding='utf-8') as file:
        for box in id_1_bboxes:
            file.write(str(box) + "\n")

    with open(os.path.join(prediction_txt_path,"id-2" + ".txt"), "w", encoding='utf-8') as file:
        for box in id_2_bboxes:
            file.write(str(box) + "\n")

    with open(os.path.join(prediction_txt_path,"bank-1" + ".txt"), "w", encoding='utf-8') as file:
        for box in bank_1_bboxes:
            file.write(str(box) + "\n")

    with open(os.path.join(prediction_txt_path,"bank-2" + ".txt"), "w", encoding='utf-8') as file:
        for box in bank_2_bboxes:
            file.write(str(box) + "\n")

    with open(os.path.join(prediction_txt_path,"xsz-1" + ".txt"), "w", encoding='utf-8') as file:
        for box in xsz_1_bboxes:
            file.write(str(box) + "\n")

    with open(os.path.join(prediction_txt_path,"xsz-2" + ".txt"), "w", encoding='utf-8') as file:
        for box in xsz_2_bboxes:
            file.write(str(box) + "\n")

    with open(os.path.join(prediction_txt_path,"xsz-3" + ".txt"), "w", encoding='utf-8') as file:
        for box in xsz_3_bboxes:
            file.write(str(box) + "\n")

    with open(os.path.join(prediction_txt_path,"jsz-1" + ".txt"), "w", encoding='utf-8') as file:
        for box in jsz_1_bboxes:
            file.write(str(box) + "\n")

    with open(os.path.join(prediction_txt_path,"jsz-2" + ".txt"), "w", encoding='utf-8') as file:
        for box in jsz_2_bboxes:
            file.write(str(box) + "\n")

def save_name():
    files = os.listdir("/Users/yanmeima/Desktop/归档/test/input/")
    with open("/Users/yanmeima/Desktop/归档/test/imgname_list.txt", "w", encoding='utf-8') as f:
        for file in files:
            if file != ".DS_Store":
                f.write(file + "\n")
# def save_name():
#     files = os.listdir("/Users/yanmeima/Desktop/add/map/original/")
#     with open("/Users/yanmeima/Desktop/add/map/imgname_list.txt", "w", encoding='utf-8') as f:
#         for file in files:
#             if file != ".DS_Store":
#                 name, ext = os.path.splitext(file)
#                 imgname = name + ".jpg"
#                 f.write(imgname + "\n")

def mAP(detpath, imagesetfile, cachedir):
    mAP = 0
    class_names_list = ['id-1', 'id-2', 'bank-1', 'bank-2', 'xsz-1', 'xsz-2', 'xsz-3', 'jsz-1', 'jsz-2']
    for classname in class_names_list:
        print(classname)
        rec, prec, ap = voc_eval(detpath, imagesetfile, classname, cachedir)
        mAP += ap

    mAP = float(mAP) / len(class_names_list)
    print("mAP:", mAP)

    return mAP



json_path = "/Users/yanmeima/Desktop/add/map/prediction/"
#detpath = "/Users/yanmeima/Desktop/add/map/ids/prediction.txt" # 预测的结果，产生的txt文件，里面是一张图片的分类别的各个检测框结果。
detpath = "/Users/yanmeima/Desktop/add/map/ids/"
imagesetfile = "/Users/yanmeima/Desktop/add/map/imgname_list.txt"  # 一个txt文件，里面是每个图片的地址，每行一个地址。
cachedir = "/Users/yanmeima/Desktop/add/map/original/"  # 缓存标注的目录,Ground Truth标注信息



if __name__ == "__main__":
    # 第一步
    #save_name()

    # 第二步
    # 格式转换：json---->txt： name  confidence  bbox
    #convert(json_path, detpath)

    # 第三步
    # # 单类别测试
    # classname = "id-1"  # 种类的名字，即类别
    # ovthresh_list = [0.5] # 0.75
    # for ovthresh in ovthresh_list:
    #     rec, prec, ap = voc_eval(detpath, imagesetfile, classname, cachedir, ovthresh)

   mAP(detpath, imagesetfile, cachedir)
