#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import os
import json

def read_jsonfile(path):
    with open(path, "r", encoding='utf-8') as f:
        return json.load(f)

def save_coco_json(instance, save_path):
    json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

# 修正shapes参数
def revise_shapes(obj):

    # 读取shapes
    old_shapes = obj['shapes']
    new_shapes = []

    # 对shapes进行修改
    for i in range(len(old_shapes)):
        shape = old_shapes[i]

        # 获取每个shape
        # 更改标签
        if shape['label'] == '0':
            shape['label'] = 'Text'
        if shape['label'] == '1':
            shape['label'] = 'Title'
        if shape['label'] == '2':
            shape['label'] = 'Seal'
        if shape['label'] == '3':
            shape['label'] = 'Handwriting'
        if shape['label'] == '4':
            shape['label'] = 'Table'
        if shape['label'] == '5':
            shape['label'] = 'Figure'

        # 更改shape_type
        shape['shape_type'] = 'polygon'

        ### 修改mask区域的点坐标
        # 获取旧的点坐标
        points = shape['points']
        point1 = points[0]
        point2 = points[1]

        # 将2点变为4点
        point_1 = [point1[0], point1[1]]
        point_2 = [point2[0], point1[1]]
        point_3 = [point2[0], point2[1]]
        point_4 = [point1[0], point2[1]]
        ## 针对线改4点，考虑到标注的如果是线的话，也是两个点，所以就一起写了
        # point_1 = [point1[0], point1[1]]
        # point_2 = [point1[0] + 1, point1[1]]
        # point_3 = [point1[0] + 2, point1[1]]
        # point_4 = [point2[0], point2[1]]

        # 将新的点坐标填入points
        shape['points'] = [point_1, point_2, point_3, point_4]

        new_shapes.append(shape)

    # 用新的shapes代替
    obj['shapes'] = new_shapes

    return obj

if __name__ == '__main__':
    # 待修改的json文件目录
    # json_dir = '/Users/yanmeima/data/detection/labelme'
    json_dir = '/home/mayanmei/ocr_project/data/train_data'
    # 修改后文件的保存路径
    # new_dir = '/Users/yanmeima/data/detection/labelme_4points'
    new_dir = '/home/mayanmei/ocr_project/data/labelme_4points'
    # 如果没有则新建一个
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    # 读取json目录里的文件
    json_files = os.listdir(json_dir)
    for file in json_files:
        # 获取json文件路径
        print(file)
        name, ext = os.path.splitext(file)
        print(ext)
        if ext == '.json':
            json_path = os.path.join(json_dir, file)
            print(json_path)
            # 读取json文件
            obj = read_jsonfile(json_path)

            # 修改json文件，将矩形的2点表达形式变为4点表达形式
            new_obj = revise_shapes(obj)

            # 新的json文件路径
            new_json = os.path.join(new_dir, file)

            # 保存新的json文件
            save_coco_json(new_obj, new_json)
