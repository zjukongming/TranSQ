import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import functools
import pickle as pkl

#from info_nce import InfoNCE
from sentence_transformers import SentenceTransformer

from torch.utils.data.distributed import DistributedSampler
from einops import rearrange
from torch.nn import MSELoss, CosineEmbeddingLoss, BCEWithLogitsLoss
from transq.modules.dist_utils import all_gather

from typing import Optional
from functools import partial
#from qubvel git
from torch.nn.modules.loss import _Loss
from transq.modules._functional import focal_loss_with_logits
from loss.ResampleLoss import ResampleLoss
#from .constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE


__all__ = ["FocalLoss"]


class SentenceCriterion(nn.Module):
    def __init__(self):
        super(SentenceCriterion, self).__init__()
        self.cos_loss = CosineEmbeddingLoss()
        #self.cls_loss = BCEWithLogitsLoss()
        #self.cls_loss = MultiLabelCategricalCE()
        
        self.cls_loss = ResampleLoss(use_sigmoid=True,reweight_func='rebalance',
                    focal=dict(focal=True, balance_param=2.0, gamma=2),
                    logit_reg=dict(neg_scale=2.0, init_bias=0.05),
                    map_param=dict(alpha=0.1, beta=10, gamma=0.2),
                    loss_weight=2.0
                    )
        
    def forward(self, sent_feats, sent_embeds, sent_num, pred_topic, idx, e_idx):
        bs = sent_feats.shape[0]
        
        #Loss of Similarity
        pred_feats = sent_feats[idx]
        tgt_feats = torch.cat([sent_embeds[i, :sent_num[i]] for i in range(bs)])
        label = torch.ones(tgt_feats.shape[0]).to(sent_feats.get_device())
        #print(pred_feats.shape, tgt_feats.shape, label.shape)
        sim_loss = self.cos_loss(pred_feats, tgt_feats, label)
        
        #Loss of Output Decision
        
        tgt_topic = torch.zeros_like(pred_topic, dtype=torch.float32, device=pred_topic.get_device())
        tgt_topic[e_idx] = 1
        cls_loss = self.cls_loss(pred_topic, tgt_topic)
        
        losses = sim_loss+1*cls_loss
        #print(sim_loss, cls_loss.mean())
        #print(cls_loss)
        #print("sent_loss", losses.shape)
        return losses, sim_loss



def compute_sent_loss(batch, phase="val"):
    
    #print("infer:", batch)
    sent_feats  = batch["sent_feats"]
    sent_embeds = batch["sent_embeds"]
    sent_num    = batch["sent_num"]
    topic_preds = batch["topic_preds"]          #(batch_size, topic_num, 1)
    indices  = batch["indices"]              
    expand_indices = batch["expand_indices"]   

    sent_criterion = SentenceCriterion()    
    sent_loss, sim_loss = sent_criterion(sent_feats, sent_embeds, sent_num, topic_preds, indices, expand_indices)
    return sent_loss, sim_loss

def compute_mimic(pl_module, batch):
    phase = "train" if pl_module.training else "val"
    infer = pl_module.infer(batch, phase, mask_text=False, mask_image=False)
    #print(infer)
    #loss_
    
    #text_loss, sent_loss = compute_text_loss(infer, phase)
    #print(text_loss.data, sent_loss.data)
    #loss = text_loss+sent_loss

    sent_loss, sim_loss = compute_sent_loss(infer)
    loss = sent_loss.mean()
    #print(loss.shape)

    ret = {
        "path": infer["path"],
        "attn": infer["attn"].cpu().detach().numpy(),
        "patch": infer["patch_index"],        
        "ids":  infer["ids"],
        "ids":  infer["ids"],
        "loss": loss,
    }
    
    #pl_module.log(f"mimic/{phase}/loss", loss)

    if phase!="train":
        #print("text_loss:{}\t sent_loss:{}".format(text_loss, sent_loss))
        
        #score = getattr(pl_module, f"{phase}_mimic_score")(
        #    pl_module.tokenizer, infer["text_logits"] ,infer["text_ids"]
        #)
        #print("sent_loss:{}".format(sent_loss))
        #ret_score = getattr(pl_module, f"{phase}_mimic_score")(
        #        pl_module.tokenizer, infer["sent_feats"] ,infer["text_ids"]
        #    )
        #score = getattr(pl_module, f"{phase}_mimic_score")(
        #        infer["topic_preds"], infer["topic_label"]
        #    )
        score = getattr(pl_module, f"{phase}_mimic_score")(
               loss.mean(),
            )
        #pl_module.log(f"mimic/{phase}/score", ret_score)
        
        ret_result = infer["ret_result"]
        #tokenizer = infer["tokenizer"]

        bleu_set = []
        tar_set = []
        trans_set = []
        for sent_ret, sent_tar in zip(ret_result[0],ret_result[1]):
            #print("===")
            #print("target:", sent_tar)
            #print("pred:", sent_ret)
            #print("retrieval sim", s)
            
            #trans_list = sent_ret.lower().split()
            #tar_list = sent_tar.lower().split()
            #trans = [trans_list]
            #ref = [[tar_list]]
            
            trans = [sent_ret.lower()]
            ref = [[sent_tar.lower()]]
            
            tar_set.append(sent_tar.lower())
            trans_set.append(sent_ret.lower())
            """
            if "" in trans_list:
                trans_list = trans_list.remove("")
            if "" in tar_list:
                tar_list = tar_list.remove("")
            """

            bleu_1 = getattr(pl_module, f"{phase}_mimic_bleu1")(trans, ref).cpu().numpy() 
            bleu_2 = getattr(pl_module, f"{phase}_mimic_bleu2")(trans, ref).cpu().numpy()   
            bleu_3 = getattr(pl_module, f"{phase}_mimic_bleu3")(trans, ref).cpu().numpy()   
            bleu_4 = getattr(pl_module, f"{phase}_mimic_bleu4")(trans, ref).cpu().numpy()

            #bleu_1, bleu_2, bleu_3, bleu_4 = 0,0,0,0
            bleu_set.append([bleu_1, bleu_2, bleu_3, bleu_4])
            #print(bleu_1.cpu().numpy(),bleu_2.cpu().numpy(bleu_set),bleu_3.cpu().numpy(),bleu_4.cpu().numpy())   
            #print(bleu_1,bleu_2,bleu_3,bleu_4)

        scores = getattr(pl_module, f"{phase}_mimic_mrg_score")(trans_set, tar_set)
        ret_result = [ret_result[0],ret_result[1],ret_result[2],ret_result[3],ret_result[4],ret_result[5],ret_result[6],bleu_set]
        
    else:
        ret_result = None         
    return ret, ret_result

def compute_iuxray(pl_module, batch):
    phase = "train" if pl_module.training else "val"
    infer = pl_module.infer(batch, phase, mask_text=False, mask_image=False)
    #print(infer)
    #loss_
    
    #text_loss, sent_loss = compute_text_loss(infer, phase)
    #print(text_loss.data, sent_loss.data)
    #loss = text_loss+sent_loss

    sent_loss, sim_loss = compute_sent_loss(infer)
    loss = sent_loss.mean()
    #print(loss.shape)

    ret = {
        "path": infer["path"],
        "attn": infer["attn"].cpu().detach().numpy(),
        "patch": infer["patch_index"],        
        "ids":  infer["ids"],
        "loss": loss,
    }
    
    #pl_module.log(f"mimic/{phase}/loss", loss)

    if phase!="train":
        #print("text_loss:{}\t sent_loss:{}".format(text_loss, sent_loss))
        
        #score = getattr(pl_module, f"{phase}_mimic_score")(
        #    pl_module.tokenizer, infer["text_logits"] ,infer["text_ids"]
        #)
        #print("sent_loss:{}".format(sent_loss))
        #ret_score = getattr(pl_module, f"{phase}_mimic_score")(
        #        pl_module.tokenizer, infer["sent_feats"] ,infer["text_ids"]
        #    )
        #score = getattr(pl_module, f"{phase}_mimic_score")(
        #        infer["topic_preds"], infer["topic_label"]
        #    )
        score = getattr(pl_module, f"{phase}_iuxray_score")(
               loss.mean(),
            )
        #pl_module.log(f"mimic/{phase}/score", ret_score)
        
        ret_result = infer["ret_result"]
        #tokenizer = infer["tokenizer"]
        #print(ret_result)

        bleu_set = []
        tar_set = []
        trans_set = []
        for sent_ret, sent_tar, s in zip(ret_result[0],ret_result[1],ret_result[2]):
            #print("===")
            #print("target:", sent_tar)
            #print("pred:", sent_ret)
            #print("retrieval sim", s)
            
            #trans_list = sent_ret.lower().split()
            #tar_list = sent_tar.lower().split()
            #trans = [trans_list]
            #ref = [[tar_list]]
            
            trans = [sent_ret.lower()]
            ref = [[sent_tar.lower()]]
            
            tar_set.append(sent_tar.lower())
            trans_set.append(sent_ret.lower())
            """
            if "" in trans_list:
                trans_list = trans_list.remove("")
            if "" in tar_list:
                tar_list = tar_list.remove("")
            """

            bleu_1 = getattr(pl_module, f"{phase}_iuxray_bleu1")(trans, ref).cpu().numpy()   
            bleu_2 = getattr(pl_module, f"{phase}_iuxray_bleu2")(trans, ref).cpu().numpy()   
            bleu_3 = getattr(pl_module, f"{phase}_iuxray_bleu3")(trans, ref).cpu().numpy()   
            bleu_4 = getattr(pl_module, f"{phase}_iuxray_bleu4")(trans, ref).cpu().numpy()
            
            #bleu_1, bleu_2, bleu_3, bleu_4 = 0,0,0,0
            bleu_set.append([bleu_1, bleu_2, bleu_3, bleu_4])
            #print(bleu_1.cpu().numpy(),bleu_2.cpu().numpy(bleu_set),bleu_3.cpu().numpy(),bleu_4.cpu().numpy())   
            #print(bleu_1,bleu_2,bleu_3,bleu_4)

            pl_module.log(f"iuxray/{phase}/score", s)  

        scores = getattr(pl_module, f"{phase}_iuxray_mrg_score")(trans_set, tar_set)
        ret_result = [ret_result[0],ret_result[1],ret_result[2],ret_result[3],bleu_set]
    else:
        ret_result = None         
    return ret, ret_result

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

