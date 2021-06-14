# -*- coding: utf-8 -*-
import os
import argparse

from sklearn.model_selection import train_test_split
from flyai.framework import FlyAI
from flyai.data_helper import DataHelper
from path import MODEL_PATH, DATA_PATH


from flyai.utils import remote_helper

import logging
import torch
print(torch.__version__)
import pandas as pd
from data_helper import Config
from fast_bert.metrics import accuracy
from fast_bert.learner_cls import BertLearner
from fast_bert.data_cls import BertDataBunch
import time

# 项目的超参，不使用可以删除
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=1 , type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=4, type=int, help="batch size")
opt = parser.parse_args()


datapath = DATA_PATH+"/MedicalClass"
print(datapath)
# labelpath = "./"
labelpath = DATA_PATH
logger = logging.getLogger()

online = True


args = Config(
    model_type='bert',
    model_dir=DATA_PATH + "/model/bert-base-chinese",
    max_seq_len=256,
    max_lr=5e-5,
    bs=24,
    epochs=8,
)


class Main(FlyAI):
    def download_data(self):
        # 下载数据
        data_helper = DataHelper()
        data_helper.download_from_ids("MedicalClass")
        print('=*=数据下载完成=*=')

    def deal_with_data(self):
        ''' 处理数据，没有可不写。 :return: '''
        # 加载数据

        all_data = pd.read_csv(os.path.join(
            DATA_PATH, 'MedicalClass/train.csv'))
        all_data = all_data.dropna()
        all_data["combined"] = all_data["title"] + all_data["text"]
        del all_data['title']
        del all_data['text']
        train_data, valid_data = train_test_split(
            all_data, test_size=0.0005, random_state=2020, shuffle=True)
        train_data.to_csv(DATA_PATH+"/MedicalClass/t.csv", index=0)
        valid_data.to_csv(DATA_PATH+"/MedicalClass/v.csv", index=0)

    def prepare(self):
        print("#准备工作！！！！！！！！！！！！！")
        device_cuda = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        metrics = [{'name': 'accuracy', 'function': accuracy}]
        databunch = BertDataBunch(datapath, labelpath,
                                  tokenizer=args.model_dir,
                                  train_file='t.csv',
                                  val_file='v.csv',
                                  label_file='label_utf8.csv',# 这个文件在那理
                                  text_col='combined',
                                  label_col='label',
                                  batch_size_per_gpu=args.bs,
                                  max_seq_length=args.max_seq_len,
                                  multi_gpu=False,
                                  multi_label=False,
                                  model_type=args.model_type)

        self.learner = BertLearner.from_pretrained_model(
            databunch,
            pretrained_path=args.model_dir,
            metrics=metrics,
            device=device_cuda,
            logger=logger,
            output_dir=MODEL_PATH,
            grad_accumulation_steps=1,
            warmup_steps=28800,
            multi_gpu=False,
            is_fp16=False,
            multi_label=False,
            logging_steps=2000)
        print("准备开始训练！！！！！！！！！！！！")

    def train(self):
        print("准备开始训练！！！！！！！！！！！！")
        a = time.time()
        self.learner.fit(epochs=args.epochs,
                         lr=args.max_lr,
                         validate=True, 	# Evaluate the model after each epoch
                         schedule_type="warmup_linear",
                         optimizer_type="adamw")
        b = time.time()
        print(b-a)
        self.learner.save_model()


if __name__ == '__main__':
    main = Main()
    main.download_data()
    main.deal_with_data()
    main.prepare()
    main.train()
    exit(0)