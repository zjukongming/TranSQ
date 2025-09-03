import mmcv
import numpy as np
import json
import torch
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib as mpl
#plt.switch_backend('agg')
import seaborn as sns
import os

cmap = mpl.cm.get_cmap("viridis", 256).colors[:,:-1]


def generate_attn_map(attn, patch):
    h, w = patch[:,0].max()+1, patch[:,1].max()+1
    attn_map = np.zeros([h, w])
    for idx, i in enumerate(patch):
        x = i[0]
        y = i[1]
        attn_map[x,y] = attn[idx]

    return attn_map, (h,w)

def generate_new_image(image, attn, patch_shape):
    h,w = patch_shape
    
    heatmap = np.zeros_like(image)
    
    h_step, w_step = image.shape[:2]
    h_step, w_step = h_step/h, w_step/w
    for i in range(h):
        for j in range(w):
            h_s = int(i*h_step)
            h_e = int((i+1)*h_step)
            w_s = int(j*w_step)
            w_e = int((j+1)*w_step)
            heatmap[h_s:h_e,w_s:w_e, :] = np.array(cmap[min(int(attn[i,j]*256), 255)])*256
            #print(cmap[min(int(attn[i,j]*256), 255)]*256)
            #image[h_s:h_e,w_s:w_e, :] = image[h_s:h_e,w_s:w_e, :] *attn[i,j]
    return np.array(heatmap*0.5+image*0.5, dtype=np.uint8)

def makedir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

data = mmcv.load("./ret_logs/test_log.pkl")
keys = data.keys()

image_path = "/mnt/vdb1/Data/mimic_cxr/images" 
output_path = "./visualization"
makedir(output_path)
for k in keys:
    
    t_data = data[k]

    bleu_4 = t_data["bleu"][3]
    if len(t_data["pred_topic"])!=7 or bleu_4 < 0.2:
        continue

    #print(t_data)
    #print("path", t_data["path"])
    #print("patch", t_data["patch"])
    pid = k
    path = os.path.join(output_path, pid)
    makedir(path)

    image = io.imread(os.path.join(image_path, t_data["path"]))
    
    patch = t_data["patch"]
    attn = t_data["attn"][:,:,1:].mean(axis=0)
    #print(attn1.shape)
    #print(attn2.shape)

    io.imsave(os.path.join(path, "origion.png"), image)

    #log

    log = dict()
    sent_set = t_data["pred"].split(" . ")
    log["gt"] = t_data["targ"]
    for idx, i in enumerate(t_data["pred_topic"]):
        log[i] = sent_set[idx]
    with open(os.path.join(path, "log.json"), "w") as f:
        json.dump(log, f)

    #visualizations
    for idx in t_data["pred_topic"]:
        max_attn = attn[idx].max()
        min_attn = attn[idx].min()

        t_attn = (attn[idx]-min_attn)/(max_attn-min_attn)
        #print(t_attn1)
        #print(t_attn2)
        attn_map, patch_shape = generate_attn_map(t_attn, patch)
        #print(attn1_map)
        #print(attn2_map)
        new_image = generate_new_image(image, attn_map, patch_shape)
        
        io.imsave(os.path.join(path, "{}.png".format(idx)), new_image)
    #draw new
    #exit()