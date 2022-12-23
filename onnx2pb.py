# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import os
import onnx
import torch
import numpy as np
import tensorflow as tf
from onnx_tf.backend import prepare

def convert(pb_output_path, onnx_model_path):
    """
    convert onnx model to tensorflow pb model
    """
    onnx_model = onnx.load(onnx_model_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(pb_output_path)

if __name__ == '__main__':

    pb_output_path = ".\\pb_model\\hybridnet\\1"
    onnx_model_path = ".\\onnx\\onnx_model\\hybridnet.onnx"
    convert(pb_output_path, onnx_model_path)
