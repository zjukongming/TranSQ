# TranSQ

MICCAI 22 accepted paper [TranSQ: Transformer-based Semantic Query for Medical Report Generation](https://link.springer.com/chapter/10.1007/978-3-031-16452-1_58) for medical report generation.

## Step 1：Pre-process

分别运行preprocess文件夹下data_preprocess.py、generate_CE.py和generate_sentence_gallery.py，以得到预处理样本、初始化类别统计和检索句子库。

## Step 2：Train

修改transq/config.py配置文件，运行run.py文件。

## Step 3：Post-process & Evaluate

1、使用训练得到的模型，测试训练集，为统计语义向量平均位置提供数据；
2、运行evaluation/NLG_eval/test_from_json.py文件，以计算训练集结果中提及语句的平均位置顺序，并对测试集结果进行重排序，计算最终指标。