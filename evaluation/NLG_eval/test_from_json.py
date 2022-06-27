import json
import mmcv
import numpy as np
import sys
sys.path.append("../..")

from preprocess.tokenizer import Tokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor import Meteor
from pycocoevalcap.rouge import Rouge

train_log_path = "../../ret_logs/train_log.json"
test_log_path  = "../../ret_logs/test_log_v28_backup.json"

json_file = open(train_log_path, "r")
log_dct = json.load(json_file)
#log_dct = mmcv.load("./ret_logs/test_log.pkl")
keys = log_dct.keys()

data_dir = "/mnt/vdb1/Data/mimic_cxr/"
threshold = 3
data_name = "mimic"
tokenizer = Tokenizer(data_dir, threshold, data_name)

pred_set = []
targ_set = []

bleu = Bleu(4)
meteor = Meteor()
rouge = Rouge()

K=50

topic_count = np.zeros(K)
topic_total = np.zeros(K)

for k in keys:
    log = log_dct[k]
    #pred = tokenizer.clean_report_mimic_cxr(log["pred"])
    #targ = tokenizer.clean_report_mimic_cxr(log["targ"])
    pred_topic = log["pred_topic"]
    targ_topic = log["gt_topic"]
    for i in range(len(targ_topic)):
        idx = targ_topic[i]
        topic_count[idx]+=i
        topic_total[idx]+=1

    #pred_set.append(pred)
    #targ_set.append(targ)

topic_avg = topic_count/topic_total

#print(topic_avg.argsort())

#json_file = open(test_log_path, "r")
#log_dct = json.load(json_file)
log_dct = mmcv.load("../../ret_logs/test_log.pkl")
keys = log_dct.keys()


for k in keys:
    log = log_dct[k]
    pred = tokenizer.clean_report_mimic_cxr(log["pred"])
    targ = tokenizer.clean_report_mimic_cxr(log["targ"])
    pred_sent_set = pred.split(".")
    pred_topic = log["pred_topic"]
    topic_sort = np.argsort(topic_avg[pred_topic])
    #print(topic_sort)
    new_sent = ""
    for i in topic_sort:
        new_sent= new_sent+pred_sent_set[i].strip()+" . "
    new_sent = tokenizer.clean_report_mimic_cxr(new_sent)
    #new_sent = pred
    pred_set.append(new_sent)
    targ_set.append(targ)


bleu_result = bleu.compute_score({i: [gt] for i, gt in enumerate(targ_set)},
                                    {i: [re] for i, re in enumerate(pred_set)},
                                    verbose=0)  

meteor_result = meteor.compute_score({i: [gt] for i, gt in enumerate(targ_set)},
                                    {i: [re] for i, re in enumerate(pred_set)})  

rouge_result = rouge.compute_score({i: [gt] for i, gt in enumerate(targ_set)},
                                    {i: [re] for i, re in enumerate(pred_set)})  
                                   

print(bleu_result[0], meteor_result[0], rouge_result[0])

