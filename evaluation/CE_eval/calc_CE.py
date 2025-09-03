import csv
from sklearn import metrics

f_pred_file = "labeled_reports_pred.csv"
f_targ_file = "labeled_reports_target.csv"

def convert_to_result(csv_list, uncertain=True):
    ret = []
    for item in csv_list:
        if item == "" or item == "0.0":
            ret.append(0)
        elif item == "-1.0":
            if uncertain==True:
                ret.append(1)
            else:
                ret.append(0)
        else: 
            ret.append(1)
    return ret

def generate_list(fname):
    f = open(fname)
    targ_reader = csv.reader(f)
    ret = []
    for row in targ_reader:
        #print(row[1:])
        #print(convert_to_result(row[1:]))
        ret = ret + convert_to_result(row[1:])
    return ret

#targ_list = generate_list(f_targ_file)
pred_list = generate_list(f_pred_file)
targ_list = generate_list(f_targ_file)

print(len(pred_list))

p = metrics.precision_score(pred_list, targ_list)
r = metrics.recall_score(pred_list, targ_list)
f1 = metrics.f1_score(pred_list, targ_list)

print(p, r, f1)



