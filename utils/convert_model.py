# coding: utf-8
"""
模型转换 h5 > pb
"""
import os

import tensorflow as tf

import keras
from keras.models import load_model
import tensorflow as tf
import os.path as osp
import os
from keras import backend as K
from m_rcnn.mask_rcnn import MaskRCNN


# 转换函数
def h5_to_pb(h5_model, output_path, out_prefix="output_", input_prefix="input_"):
    out_nodes = {}
    input_nodes = {}
    for i in range(len(h5_model.outputs)):
        # out_nodes.append(out_prefix + str(i + 1))
        # tf.identity(h5_model.output[i], out_prefix + str(i + 1))
        out_nodes[out_prefix + str(i + 1)] = \
            tf.saved_model.utils.build_tensor_info(h5_model.output[i])

    for i in range(len(h5_model.inputs)):
        # out_nodes.append(input_prefix + str(i + 1))
        # tf.identity(h5_model.output[i], input_prefix + str(i + 1))
        input_nodes[input_prefix + str(i + 1)] = \
            tf.saved_model.utils.build_tensor_info(h5_model.input[i])

        print()
    sess = K.get_session()
    # inputs = {
    #     "input_data": tf.saved_model.utils.build_tensor_info(input_images)
    # }
    # # B方案.直接输出一个整个的SparseTensor
    # output = {
    #     "output": tf.saved_model.utils.build_tensor_info(seg_maps_pred),
    # }
    convert(sess, input_nodes, out_nodes, output_path)
    # from tensorflow.python.framework import graph_util, graph_io
    # init_graph = sess.graph.as_graph_def()
    # main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)
    # graph_io.write_graph(main_graph, output_dir, name=model_name, as_text=False)


def mk_dir(path):
    if not os.path.exists(path):
        print("创建目录：", path)
        os.makedirs(path)


def convert(sess, inputs, output, output_path):
    # 保存转换好的模型目录
    # saveModDir = "models"
    saveModDir = output_path
    mk_dir(saveModDir)
    # 每次转换都生成一个版本目录
    for i in range(100000, 9999999):
        cur = os.path.join(saveModDir, str(i))
        if not tf.gfile.Exists(cur):
            saveModDir = cur
            break

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3, allow_growth=True)
    ses_config = tf.ConfigProto(gpu_options=gpu_options)
    print("模型保存目录", saveModDir)
    # 原ckpt模型
    # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # 保存转换训练好的模型
    builder = tf.saved_model.builder.SavedModelBuilder(saveModDir)

    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=inputs,
        outputs=output,
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )

    builder.add_meta_graph_and_variables(
        sess=sess,
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={  # 保存模型的方法名，与客户端的request.model_spec.signature_name对应
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
        }
    )
    builder.save()
    print("转换模型结束", saveModDir)


def main():
    # h5模型路径
    # h5_model_name = 'model.h5'
    # weight_path = "models/mask_rcnn_coco_0078.h5"
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight_path', default='models/zx_table/mask_rcnn_coco_1001.h5')
    parser.add_argument('-o', '--output_path', default='models/pb_zx_table')
    args = parser.parse_args()
    weight_path = args.weight_path
    output_path = args.output_path
    # 加载网络模型
    mask_model = MaskRCNN(train_flag=False)
    # 加载权重模型
    mask_model.load_weights(weight_path, by_name=True)

    h5_to_pb(mask_model.keras_model, output_path)

    print('model saved')


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = "3"

    main()
