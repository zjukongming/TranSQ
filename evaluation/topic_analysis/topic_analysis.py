import mmcv
import numpy as np
import csv
import seaborn as sns
import plotly.offline as plotoff
from plotly.graph_objs import *
import plotly.io as pio
import matplotlib.pyplot as plt
plt.switch_backend('Agg')


def draw(heatmap, query, labels_count, trems):

    trace1 = {
        "name": "上",
        "type": "bar",
        "x": list(range(14)),
        "y": labels_count,
        "xaxis": "x",
        "yaxis": "y2",
        "width": 0.3,
        "marker": {
            "color": "#0099CC"
        },
        "opacity": 0.5,
    }
    """
    trace2 = {
        "name": "左",
        "type": "bar",
        "x": list(range(14)),
        "y": np.random.rand(14),
        "xaxis": "x2",
        "yaxis": "y",
        "marker": {
            "color": "#0099CC"
        },
        "opacity": 0.5,
    }
    """
    trace2 = {
        "name": "右",
        "type": "bar",
        "x": query,
        "y": np.array(range(50)),
        "xaxis": "x2",
        "yaxis": "y",
        "marker": {
            "color": "#0099CC"
        },
        "opacity": 0.5,
        "orientation": "h",
    }
    trace3 = {
        "type": "heatmap",
        #"xsrc": "Python-Demo-Account:18311:4ea2d7",
        "x": np.array(range(50))-0.5,
        "y": np.array(range(14)),
        "z": heatmap,
        # "colorscale": [
        #     [0.0, "#440154"], [0.1111111111111111, "#482878"], [0.2222222222222222, "#3e4989"],
        #     [0.3333333333333333, "#31688e"], [0.4444444444444444, "#26828e"], [0.5555555555555556, "#1f9e89"],
        #     [0.6666666666666666, "#35b779"], [0.7777777777777778, "#6ece58"], [0.8888888888888888, "#b5de2b"],
        #     [1.0, "#fde725"]
        # ],
        "xaxis": "x",
        "yaxis": "y",
        "colorbar_thickness": 150,
        "colorbar_tickfont": dict(size=120)
    }
    layout = {
        #"title": "Date Plots",
        "width": 1000,
        "xaxis": {
            "anchor": "x",
            "domain": [0.0, 0.9],
            #"showticklabels": False,
            "tickvals": np.array(range(14)),
            "ticktext": terms,
            #"tickfont": "Arial",
            "tickfont": dict(family='Arial', size=150),
            #"ticklabelposition": "outside",
        },
        "yaxis": {
            "anchor": "y",
            "domain": [0.0, 0.85],
            "tickmode": "array",
            "tickfont": dict(family='Arial', size=150),
            #"showticklabels": False
            "tickvals": np.array([ 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]),
            "ticktext": np.array([45, 40, 35, 30, 25, 20, 15, 10,  5,  1]),
            #"ticklabelposition": "outside",
        },
        "height": 700,
        "xaxis2": {
            "anchor": "x2",
            "tickmode":"array",
            "domain": [0.91, 1.0],
            "tickfont": dict(family='Arial', size=120),
            #"ticklabelposition": "outside left",
            #"showticklabels": False,
        },
        "yaxis2": {
            "anchor": "y2",
            "domain": [0.87, 1.0],
            "tickfont": dict(family='Arial', size=120)
            #"showticklabels": False,
        },
        "showlegend": False,
    }
    data = [trace1, trace2, trace3]
    fig = Figure(data=data, layout=layout)
    #output_path = "1.html"
    #plotoff.plot(fig, filename=output_path)
    pio.write_image(fig, '1.png', width=10000, height=8000 )
    pio.write_json(fig, "1.json")

def raw_labels_explain(raw):
    ret = np.zeros(14)
    for idx, r in enumerate(raw):
        if r != "":
            ret[idx]=1
    if ret[1:].sum()>0:
        ret[0]=0
    return ret

def csv_analysis(csv_file, cls_seq, sorted_ids):
    #load csv_file
    result = np.zeros((50,14))
    count = np.zeros(50)

    category_count = np.zeros(14)

    with open(csv_file) as f:
        f_csv = csv.reader(f)
        for idx, row in enumerate(f_csv):
            if idx==0:
                continue
            #print(idx)
            cls_id = cls_seq[idx-1]
            raw_labels = row[1:]
            labels = raw_labels_explain(raw_labels)
            #if labels[-1]==1:
            if 50-np.argwhere(sorted_ids==cls_id)[0][0]==44:
                print(50-np.argwhere(sorted_ids==cls_id))
                print(labels)
                print(row[0])
            result[cls_id,:]=result[cls_id,:]+labels 
            category_count = category_count+labels
            count[cls_id]+=1
    return result/count[:,np.newaxis], category_count/idx
                

pkl_path = "../../ret_logs/test_log.pkl"
logs = mmcv.load(pkl_path)
keys = logs.keys()

count = np.zeros(50)

for k in keys:
    log = logs[k]
    for i in log["gt_topic"]:
        count[i]+=1

sorted_prob = np.sort(count/len(keys))
sorted_ids = np.argsort(count)

print(sorted_prob)
print(sorted_ids)

cls_seq = mmcv.load("./csv_cls_seq.pkl")["targ_cls_seq"]
#print(len(cls_seq))
heatmap, category = csv_analysis("./labeled_reports_targ_sents.csv", cls_seq, sorted_ids)
print(category)

terms = ["No Finding","Enlarged Cardiomediastinum","Cardiomegaly","Lung Lesion","Lung Opacity", "Edema", "Consolidation", 
         "Pneumonia","Atelectasis","Pneumothorax","Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]
    
#sorted_ids = [42, 47, 35, 16, 27, 33, 34, 23, 15, 37, 21, 25, 29, 48, 41,  1, 44, 10, 26,  6,  5,  9,  7,  2,
#  3, 30,  4, 17, 32, 39, 11, 31, 49, 45, 20,  0, 36,  8, 28, 19, 43, 18, 46, 13, 40, 22, 38, 12,
# 24, 14]

draw(heatmap[sorted_ids], sorted_prob, category, terms)
#np.set_printoptions(precision=2)
#np.set_printoptions(suppress=True)

"""
ax = sns.heatmap(heatmap).get_figure()
fig_path = "./1.png"
ax.set_figwidth(3)
ax.set_figheight(5)
ax.savefig(fig_path, dpi = 400)
"""