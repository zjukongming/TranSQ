import json
import mmcv
import numpy as np
import sys
import pickle as pkl
sys.path.append("../..")

from preprocess.tokenizer import Tokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge


from transq.gadgets.my_metrics import Accuracy, VQAScore, Scalar, MRGScore, MRG_Retrieval, MSEScore, TopicAccuracy, BLEUScore

train_log_path = "../../ret_logs/base/train.json"

json_file = open(train_log_path, "r")
log_dct = json.load(json_file)
#log_dct = mmcv.load("./ret_logs/test_log.pkl")
keys = log_dct.keys()

data_dir = "/big-disk/mimic_cxr/"
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
test_log_path = "../../ret_logs/gen/result.pkl"
json_file = open(test_log_path, "rb")
log_dct = pkl.load(json_file)
keys = log_dct.keys()

for k in keys:
    log = log_dct[k]
    pred = tokenizer.clean_report_iu_xray(log["pred_sentence"])
    targ = tokenizer.clean_report_iu_xray(log["targ_sentence"])
    path = log["path"]

    #sim = log["ret_sim"]
    pred_topic = log["pred_id"]
    gt_topic = log["gt_id"]

    pred_sent_set = pred.split(".")

    topic_sort = np.argsort(topic_avg[pred_topic])
    #print(topic_sort)
    new_sent = ""
    for i in topic_sort:
        new_sent= new_sent+pred_sent_set[i].strip()+" . "
    new_sent = tokenizer.clean_report_iu_xray(new_sent)
    #new_sent = pred
    
    new_targ = ""
    targ_list = targ.split('.')
    for i in range(len(gt_topic)):
        new_targ= new_targ+targ_list[i].strip()+" . "
    new_targ = tokenizer.clean_report_iu_xray(new_targ)
    """
    new_sim = []
    for i in topic_sort:
        new_sim.append(sim[i])
    """
    new_pred_topic = []
    for i in topic_sort:
        new_pred_topic.append(pred_topic[i])
    
    pred_set.append(new_sent)
    targ_set.append(new_targ)


bleu1 = BLEUScore(1)
bleu2 = BLEUScore(2)
bleu3 = BLEUScore(3)
bleu4 = BLEUScore(4)
score = MRG_Retrieval()
score.update(pred_set,targ_set)

for pred,targ in zip(pred_set,targ_set):
    bleu1.update([pred],[[targ]])
    bleu2.update([pred],[[targ]])
    bleu3.update([pred],[[targ]])
    bleu4.update([pred],[[targ]])
    

print(bleu1.compute())
print(bleu2.compute())
print(bleu3.compute())    
print(bleu4.compute())
print(score.compute())
