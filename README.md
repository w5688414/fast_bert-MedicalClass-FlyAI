# fast_bert-MedicalClass-FlyAI

## environment

+ fast-bert                          1.9.9
+ torch                              1.4.0+cu100
+ torchvision                        0.5.0
+ FlyAI                              0.7.1

## training

```
python main.py
```
会自动下载数据，自动运行哈

## inference

```
python prediction.py
```
输出预测的精度和值哈

最终的预测精度为：75%，多训练一下，调调参数会比这个精度更高哈