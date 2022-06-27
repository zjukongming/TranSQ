import os
import json
import sys
import torch
import torch.nn.functional as F
from tokenizer import Tokenizer
import pickle as pkl
import numpy as np

from sentence_transformers import SentenceTransformer

max_sent_num = 25
max_seq_length = 40
sent_emb = 768
sent_encoder = SentenceTransformer('all-mpnet-base-v2').encode


data_dir = "/mnt/vdb1/Data/mimic_cxr/"
threshold = 3
data_name = "mimic"
tokenizer = Tokenizer(data_dir, threshold, data_name)
vocab_size = tokenizer.get_vocab_size()
print("vocab_size", vocab_size)


def get_info(example):
    image_id = example["id"]
    image_path = example['image_path']
    sentences = example['report'].replace("\n"," ").split(". ")                     #按句拆分

    sent_len = min(max_sent_num-1, len(sentences))                                  #句子数
    sent_vec = sent_encoder(sentences)
    #参考聚类中心，对sent_vec进行排序，且记录类别
    #print(image_id)
    #print(sent_vec)
    sent_list = []
    
    for vec_i, sent_i in zip(sent_vec, sentences):
        vec = torch.Tensor(vec_i)#.to(device) #768*1
        report_sent = tokenizer(sent_i)
        #norm
        vec = F.layer_norm(vec, (768,))
        vec = F.normalize(vec, dim=0)
        sent_len = len(sent_i.split(" "))
        #print(vec.shape, centers.transpose(1,0).shape)
        #dist = 1-vec @ centers.transpose(1,0)
        #dist = torch.sum(torch.mul(vec-centers,vec-centers), dim=-1)                #cls*768 *768*1
        #idx  = torch.argmin(dist).cpu().item()
        #sent_list.append((idx, vec.cpu().numpy(), report_sent))
        sent_list.append((vec.cpu().numpy(), report_sent, sent_len))
        

    sent_vecs = []
    report_ids = []
    sent_lens = []
    #for idx, sent_vec, report_id in sent_list:
    for sent_vec, report_id, sent_len in sent_list:
        #sent_cls.append(idx)
        sent_vecs.append(sent_vec)
        report_ids.append(report_id)
        sent_lens.append(sent_len)
    #print(sent_cls)
    sent_seq = np.array(sent_vecs)[:sent_len, :]

    seq_length = max(min(len(report_ids), max_sent_num),1)
    

    sample = {  "image_id": image_id,
                "image_path": image_path,
                "report_ids": report_ids,       #按句拆分的token ids，其中已保证第一位为0
                #"sent_cls": sent_cls,
                "sent_seq": sent_seq,           #句向量序列，第一项为全0
                "sent_len": sent_lens,
                "seq_len": seq_length           #句子数
                }
    return sample


ann_path = "/mnt/vdb1/Data/mimic_cxr/annotation.json"
ann = json.loads(open(ann_path, 'r').read())
split_list = ["val", "test", "train"]

for split in split_list:
    fname = "./data/mimic_{}.pkl".format(split)
    f = open(fname, "wb")
    dataset_dict = []
    examples = ann[split]
    example_num = len(examples)
    for idx, example in enumerate(examples):
        if idx %1000==0:
            print(idx,"/",example_num)
        image_id = example["id"]
        sample = get_info(example)
        dataset_dict.append(sample)
        #print(sample)
    pkl.dump(dataset_dict, f)
    print(len(dataset_dict))
