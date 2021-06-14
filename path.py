# -*- coding: utf-8 -*
import sys
import os

# 训练数据的路径
DATA_PATH = os.path.join(sys.path[0], 'data', 'input')
# 模型保存的路径
MODEL_PATH = os.path.join(sys.path[0], 'data', 'output', 'model')
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)