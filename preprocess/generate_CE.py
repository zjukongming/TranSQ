import os
import json
import mmcv
import numpy as np

f = open("/mnt/vdb1/Data/mimic_cxr/annotation.json")
ann = json.load(f)
split_list = ["train", "val", "test"]

dct = ann["train"]

num = 50
avg = 6
class_freq = len(dct)/num*avg
class_neg_freq = len(dct)/num*(num-avg)

out_dct = {}
out_dct["class_freq"] = np.ones(num)*class_freq
out_dct["neg_class_freq"] = np.ones(num)*class_neg_freq

mmcv.dump(out_dct, "./class_freq.pkl")



        