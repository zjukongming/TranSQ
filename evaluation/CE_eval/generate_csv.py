import mmcv
import json
import csv

fname = "./ret_logs/test_log_v28_train.json"
with open(fname, "r") as f:
    dct = json.load(f)

keys = dct.keys()

in_sent = True
if in_sent:
    f_pred = open("pred_reports_sent_train_0.csv", "w")
    f_targ = open("targ_reports_sent_train_0.csv", "w")
else:
    f_pred = open("pred_reports.csv", "w")
    f_targ = open("targ_reports.csv", "w")

w_pred = csv.writer(f_pred)
w_targ = csv.writer(f_targ) 

ret_set, tar_set = [], []
ret_ids, tar_ids = [], []
pred_cls_seq, targ_cls_seq = [], []

count = 0
f_id = 0
for k in keys:
    if count>=200000:
        print("output:", f_id)
        w_pred.writerows(ret_set)
        w_targ.writerows(tar_set)
        f_pred.close()
        f_targ.close()
        f_id+=1
        f_pred = open("pred_reports_sent_train_{}.csv".format(f_id), "w")
        f_targ = open("targ_reports_sent_train_{}.csv".format(f_id), "w")
        w_pred = csv.writer(f_pred)
        w_targ = csv.writer(f_targ)
        ret_set, tar_set = [], []
        count =0 
    log = dct[k]
    if in_sent ==False:
        ret_set.append([log["pred"]])
        tar_set.append([log["targ"]])
    else:
        pred_sents = log["pred"].split(" . ")
        for idx, s in enumerate(pred_sents):
            pred_cls_seq.append(log["pred_topic"][idx])
            ret_set.append([s])
        targ_sents = log["targ"].split(" . ")
        for idx, s in enumerate(targ_sents):
            if len(s)<=3:
                continue
            targ_cls_seq.append(log["gt_topic"][idx])
            tar_set.append([s])
            count+=1
            
w_pred.writerows(ret_set)
w_targ.writerows(tar_set)

if in_sent:
    mmcv.dump({"pred_cls_seq":pred_cls_seq, "targ_cls_seq":targ_cls_seq}, "csv_cls_seq_train.pkl")

f_pred.close()
f_targ.close()


