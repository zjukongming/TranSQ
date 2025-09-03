import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import transq.modules.vision_transformer as vit
import numpy as np
import pickle as pkl
import json
import mmcv
import torchvision.models as models
from einops import rearrange
from timm.models.layers import StdConv2dSame, DropPath, to_2tuple, trunc_normal_

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from transq.modules import heads, objectives, mrg_utils
from scipy.optimize import linear_sum_assignment

class TransformerSQ_CNN(pl.LightningModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.epoch_count = -1
        self.ret_result_file = open("./ret_logs/ret_test.txt", "w")

        self.save_hyperparameters()
        self.tokenizer = tokenizer
        #self.vocab_size = vocab_size
        self.sent_len = config["max_sent_num"]
        self.text_size = config["max_text_len"]
        self.image_size = config["max_image_len"]
        """
        bert_config = BertConfig(
            #vocab_size=config["vocab_size"],
            vocab_size=7863,
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )
        """
        self.dataset_name=self.hparams.config["datasets"]
        #self.fuse = nn.Linear(290,145)
        #self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        hs = self.hparams.config["hidden_size"] 
        #self.text_embeddings = BertEmbeddings(bert_config)
        #self.text_embeddings.apply(objectives.init_weights)     

        self.pos_embeddings = vit.PositionalEncoding(config["hidden_size"], 0.1, self.sent_len)
        self.pos_embeddings.apply(objectives.init_weights)               

        self.pos_embeddings_2 = vit.PositionalEncoding(config["hidden_size"], 0.1, self.text_size)
        self.pos_embeddings_2.apply(objectives.init_weights)

        self.image_type_embedding = nn.Embedding(2, hs)
        self.image_type_embedding.apply(objectives.init_weights)

        self.vis_dropout = nn.Dropout(0.1)

        f = open("./preprocess/data/iuxray_sentence_gallery.pkl", "rb")
        #self.gallery = Gallery()
        gallery = pkl.load(f)
        self.sentence_vectors = gallery.sentence_vectors
        vecs_mean = self.sentence_vectors.mean(axis=1)[:,np.newaxis]
        vec_std  = np.sqrt(self.sentence_vectors.var(axis=1)+1e-6)[:, np.newaxis]
        self.sentence_vectors = (self.sentence_vectors-vecs_mean)/vec_std

        sent_vecs_norm = np.linalg.norm(x = self.sentence_vectors, ord=2, axis = 1, keepdims = True)
        self.sent_vecs_norm = sent_vecs_norm.clip(min=1e-7).reshape(-1,1)
        self.sent_vects = torch.Tensor(self.sentence_vectors/self.sent_vecs_norm)    
        self.sentence_gallery = gallery.sentence_gallery     

        self.semantic_query_num = config["semantic_query_num"]
        self.semantic_query = nn.Embedding(self.semantic_query_num, hs)
        self.classification_query = nn.Embedding(self.semantic_query_num, hs)
        self.train_select_pos_count = np.zeros(self.semantic_query_num)
        self.train_select_neg_count = np.zeros(self.semantic_query_num)
        self.test_select_pos_count = np.zeros(self.semantic_query_num)
        self.test_select_neg_count = np.zeros(self.semantic_query_num)
        self.path_graph = np.zeros((self.semantic_query_num+2, self.semantic_query_num+2))

        #self.graph = mmcv.load("path_graph_v28_test.pkl")
        #self.graph = self.graph/(self.graph.sum(0)+1e-7)[np.newaxis,:]      #入度
        #self.graph = self.graph/(self.graph.sum(1)+1e-7)[:,np.newaxis]      #出度

        self.test_select_count = np.zeros(self.semantic_query_num)

        self.test_log = dict()


        self.topic_proj = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, hs),
            )  
        self.topic_clas = nn.Sequential(
                nn.Linear(hs, hs),
                #nn.Linear(2*hs, hs),
                nn.LayerNorm(hs),
                nn.GELU(),
                nn.Linear(hs, self.semantic_query_num),
                #nn.Linear(hs, 1)
            )
        self.topic_proj.apply(objectives.init_weights)
        self.topic_clas.apply(objectives.init_weights)

        self.topic_clas2 = nn.Sequential(
                nn.Linear(hs, hs),
                #nn.Linear(2*hs, hs),
                nn.LayerNorm(hs),
                nn.GELU(),
                #nn.Linear(hs, self.semantic_query_num),
                nn.Linear(hs, 1)
            )
        self.topic_clas2.apply(objectives.init_weights)        

        
        if self.hparams.config["load_path"] != "":
            print("pretrained: True")
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:
            print("pretrained: False")
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )

        resnet=models.resnet50(pretrained=False)
        self.res_features = nn.Sequential(*list(resnet.children())[:-2])
        self.res_features.add_module("conv_1x1",nn.Conv2d(2048, 768, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False))
        #self.load_state_dict(torch.load()['model'])
        if (self.hparams.config["load_path"] == "" and not self.hparams.config["test_only"]):
            self.load_state_dict({k.replace('module.',''):v for k,v in torch.load('/fast-disk/kongming/Code/TranSQ-iuxray/pretrained-models/MedKLIP.pth')['model'].items()}, strict=False)


        """
        densenet=models.densenet121(pretrained=True)
        self.res_features = nn.Sequential(*list(densenet.children())[:1])
        self.res_features.add_module("conv_1x1",nn.Conv2d(1024, 768, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False))
        
        """
        """
        resnet=models.resnet50(pretrained=True)
        self.res_features = nn.Sequential(*list(resnet.children())[:-2])
        self.res_features.add_module("conv_1x1",nn.Conv2d(2048, 768, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False))
        """
        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"])
            self.itm_score.apply(objectives.init_weights)

        if config["loss_names"]["mpp"] > 0:
            self.mpp_score = heads.MPPHead(bert_config)
            self.mpp_score.apply(objectives.init_weights)

        # ===================== Downstream ===================== #
        if (self.hparams.config["load_path"] != "" and not self.hparams.config["test_only"]):
            print(self.hparams.config["test_only"])
            print("Load pretrained model from {}".format(self.hparams.config["load_path"]))
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            
            #for k,v in self.named_parameters():
            #    if not "topic_clas" or not "blocks_cls" in k:
            #        v.requires_grad = False

            #for k in self.parameters():
            #    k.requires_grad=False
            
            #self.topic_clas.requires_grad = True

        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.rank_output = nn.Linear(hs, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_score.parameters():
                p.requires_grad = False

        #if self.hparams.config["loss_names"]["mimic"] > 0:
            #vs = self.hparams.config["vqav2_label_size"]
            
            #self.mimic_score=metrics.compute_scores

        mrg_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            print("Load Pretrained Model From", self.hparams.config["load_path"])
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            print(state_dict.keys())
            print(self.state_dict().keys())
            self.load_state_dict(state_dict, strict=False)
    
    def matcher(self, sent_feats, topic_preds, sent_embeds, sent_num, prob_mask=False):
        bs, num_query, _ = sent_feats.shape                                         # sent_feats = (b, n_q, dim)
        out_cls   = topic_preds.flatten(0,1).sigmoid()                              # (b*n_q)
        out_feats = F.normalize(sent_feats.flatten(0,1), dim=1)                     # (b*n_q, dim)
        
        tgt_feats = torch.cat([sent_embeds[i, :sent_num[i]] for i in range(bs)])    # (m, dim)
        tgt_cls   = torch.zeros(tgt_feats.shape[0], dtype=torch.long)               # m
        #print("out_feats", out_feats.shape, "tgt_feats", tgt_feats.shape)
        cost_pick = -out_cls.unsqueeze(1)[:, tgt_cls]                               # 被选中则cost = -prob
        cost_sim = torch.cdist(out_feats, tgt_feats, p=2)                           # (b*n_q, m)
        
        if prob_mask == True:
            out_cls_mask = (out_cls<=0.3)
            cost_sim[out_cls_mask] = 100
        
        C = cost_sim + 0.5*cost_pick
        C = C.view(bs, num_query, -1).cpu().detach()                                # (b, n_q, m)
        #print(C[0,:, 0].topk(5, largest=False))
        #print(sent_num)
        #print(C.shape)
        indices = [linear_sum_assignment(c[i].transpose(1,0)) for i, c in enumerate(C.split(list(sent_num.cpu().numpy()), -1))] 
        #print(indices)
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for j, i in indices]

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        #print(indices)
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def generate_ret_result(self, sent_feats, text_ids, indices, gt_indices, topic_preds, batch_size, device):
        batch_indices, src_indices = indices
        gt_batch_indices, gt_src_indices = gt_indices
        sim_set = []
        sent_ret = []
        sent_targ = []
        topic_prob = []
        topic_prob_gt = []
        for i in range(batch_size):
            idx = src_indices[(batch_indices==i).nonzero().view(-1)]
            gt_idx = gt_src_indices[(gt_batch_indices==i).nonzero().view(-1)]
            #idx = self.idx_path_search(idx)
            #idx = self.idx_path_search_v2(sent_feats[i], idx, topic_preds[i])
            sent_feat_t = sent_feats[i, idx]
            topic_pred_t = topic_preds[i, idx].sigmoid().cpu().numpy()
            topic_gt_t = topic_preds[i, gt_idx].sigmoid().cpu().numpy()
            #print(idx)
            #print(sent_feat_t)
            sent_feat_t = F.normalize(sent_feat_t, dim=-1).cpu()
            #print(sent_feat_t.shape)
            sim = sent_feat_t @ self.sent_vects.transpose(1,0)
            #print(sim.shape)
            max_idx = torch.argmax(sim, dim=1).to(device)
            #print(max_idx.shape)
            sim_t=0
            sent_ret_t=""
            for j in range(len(max_idx)):
                #print(max_idx[j])
                sim_t += sim[j][max_idx[j]].cpu().item()
                pred_sent_t = self.sentence_gallery[max_idx[j]]
                pred_sent_t = pred_sent_t.strip(".")+" ."
                sent_ret_t = sent_ret_t + pred_sent_t + " "
            sent_targ_t = self.tokenizer.decode_batch(text_ids[i, :, 1:].int().cpu().numpy())
            sent_targ_str = ""
            for j in sent_targ_t:
                if j!="":
                    sent_targ_str = sent_targ_str+j+" "
            sent_ret_t = sent_ret_t.strip()
            sent_targ_str = sent_targ_str.strip()

            sent_ret_t = self.tokenizer.clean_report_iu_xray(sent_ret_t)
            sent_targ_str = self.tokenizer.clean_report_iu_xray(sent_targ_str)
            
            sim_set.append(sim_t/(len(max_idx)+1e-7))
            sent_ret.append(sent_ret_t)
            sent_targ.append(sent_targ_str)
            topic_prob.append(((topic_pred_t, idx), (topic_gt_t, gt_idx)))
            
        
        ret_result = [sent_ret, sent_targ, sim_set, topic_prob]
        return ret_result
    
    def select_indices_v2(self, sent_feats, indices=None, indices_max=None, batch_size=64, device="cuda:0"):
        batch_set = []
        src_set = []
        #new_indices = []
        for i in range(batch_size):
            sent_feats_t = sent_feats[i]                                    #(query_num, dim)
            if indices!=None:
                pre_select_idxs = indices[1][indices[0]==i]
                sent_feats_t = sent_feats_t[pre_select_idxs]
                if indices_max!=None:
                    first_select = (pre_select_idxs==indices_max[i]).nonzero()[0]
                else:
                    first_select = None

            sent_feats_t = F.normalize(sent_feats_t, dim=-1).cpu()   
            ret = sent_feats_t @ self.sent_vects.transpose(1,0)             #(query_num, gallery_len)
            max_value, max_idx = torch.max(ret, dim=1)                      #(query_num)
            max_value, max_idx = max_value.to(device), max_idx.to(device)
            
            sent_retrival = self.sent_vects[max_idx].to(device)             #(query_num, dim)
            
            info_matrix = 1-(sent_retrival @ sent_retrival.transpose(1,0))  #(query_num, query_num)
            #info_matrix = info_matrix * max_value                           #relate = distance*similarity
            info_matrix = info_matrix 
            #if i==0:
            #    print(info_matrix)
            #    print(max_value)
            select_set = []
            if first_select != None:
                select = first_select
            else:    
                select = torch.argmax(max_value)
            select_set.append(select)
            set_dist = info_matrix[select]
            batch_set.append(i)
            src_set.append(pre_select_idxs[select])
            while set_dist.max()>0.2:                                      #valuable enough for introducing
                select = torch.argmax(set_dist)
                select_set.append(select)
                set_dist = torch.min(set_dist, info_matrix[select])
                batch_set.append(i)
                src_set.append(pre_select_idxs[select])
            #new_indices.append((torch.as_tensor(src_set, dtype=torch.int64), torch.as_tensor(batch_set, dtype=torch.int64)))
            
        #indices = [(torch.as_tensor(batch_set[i], dtype=torch.int64), torch.as_tensor(src_set[i], dtype=torch.int64)) for i in range(len(batch_set))]
        
        indices = (torch.LongTensor(batch_set), torch.LongTensor(src_set))    
        
        return indices

    def select_indices(self, sent_feats, batch_size=64, device="cuda:0"):
        batch_set = []
        src_set = []
        for i in range(batch_size):
            sent_feats_t = sent_feats[i]                                    #(query_num, dim)
            sent_feats_t = F.normalize(sent_feats_t, dim=-1).cpu()   
            ret = sent_feats_t @ self.sent_vects.transpose(1,0)             #(query_num, gallery_len)
            max_value, max_idx = torch.max(ret, dim=1)                      #(query_num)
            max_value, max_idx = max_value.to(device), max_idx.to(device)
            
            sent_retrival = self.sent_vects[max_idx].to(device)             #(query_num, dim)
            
            info_matrix = 1-(sent_retrival @ sent_retrival.transpose(1,0))  #(query_num, query_num)
            info_matrix = info_matrix * max_value                           #relate = distance*similarity
            #if i==0:
            #    print(info_matrix)
            #    print(max_value)
            select_set = []
            select = torch.argmax(max_value)
            select_set.append(select)
            set_dist = info_matrix[select]
            batch_set.append(i)
            src_set.append(select)
            while set_dist.max()>0.3:                                      #valuable enough for introducing
                select = torch.argmax(set_dist)
                select_set.append(select)
                set_dist = torch.min(set_dist, info_matrix[select])
                batch_set.append(i)
                src_set.append(select)

        indices = (torch.LongTensor(batch_set), torch.LongTensor(src_set))    
        return indices

    def expand_indices(self, sent_feats, indices, batch_size):
        batch_indices, src_indices = indices
        sent_feats = F.normalize(sent_feats, dim=-1)
        new_src = []
        new_batch = []
        for i in range(batch_size):
            idx = src_indices[(batch_indices==i).nonzero().view(-1)]
            sent_feats_t = sent_feats[i, idx]
            sim = (sent_feats[i] @ sent_feats_t.transpose(1,0)).max(-1).values
            #print(idx)
            src = (sim>0.9).nonzero().view(-1).cpu()
            #print(src)
            new_src.append(src) 
            new_batch.append(torch.full_like(src, i)) 
        new_batch_indices = torch.cat(new_batch, dim=0)
        new_src_indices = torch.cat(new_src, dim=0)
        #print(new_batch_indices)
        #print(new_src_indices)
        
        return (new_batch_indices, new_src_indices)

    def infer(
        self,
        batch,
        phase = "val",
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):

        image_ids = batch["image_id"]
        image_path = batch["path"]
        text_ids = batch["report_ids"].long()
        #print(image1_path)
        #print(image2_path)

        #img = batch["image"]

        #sentence embedding
        sent_embeds = batch["sent_seq"].float()
        #sent_masks = batch["sent_mask"]
        sent_num = batch["seq_len"]

        length = torch.max(batch["seq_len"]).cpu().data

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        batch_size, sent_length, feat_dim = sent_embeds.size()

        ## Main      
        # Step 1: Visual Extractor:
        #print("step 1")

        image1=batch["image"][:, 0]
        image2=batch["image"][:, 1]
        x1=self.res_features(image1)
        x2=self.res_features(image2)
        #print(x1.size())   [64, 768, 12, 12]
        (
            image1_embeds,
            pos_embeds,
            image1_masks,
            patch_index
        )=self.transformer.create_pos_embed_patch_index(
            x1,
            max_image_len=self.hparams.config["max_image_len"]
        )
        #print("image1_embeds",image1_embeds.size())        [64, 145, 768]
        #print("pos_embeds",pos_embeds.size())              [64, 145, 768]
        (
            image2_embeds,
            pos_embeds,
            image2_masks,
            patch_index
        )=self.transformer.create_pos_embed_patch_index(
            x2,
            max_image_len=self.hparams.config["max_image_len"]
        )
        
        image1_embeds, image2_embeds = (
            image1_embeds + self.image_type_embedding(torch.zeros_like(image1_masks)),
            image2_embeds
            + self.image_type_embedding(
                torch.full_like(image2_masks, image_token_type_idx)
            ),
        )
        
        vis_embeds1=image1_embeds
        vis_embeds2=image2_embeds
        vis_embeds=torch.cat((vis_embeds1, vis_embeds2), dim=1)
        semantic_query = self.semantic_query(torch.arange(self.semantic_query_num).to(device))
        x = semantic_query.repeat(batch_size, 1, 1)
        for i, blk in enumerate(self.transformer.blocks_topic):
            x, _attn = blk(x, torch.cat((vis_embeds1+0+pos_embeds, vis_embeds2+1+pos_embeds), dim=1), (vis_embeds))
        x = self.transformer.norm_sent(x)
        sent_feats = self.topic_proj(x).float()
        topic_preds = self.topic_clas2(x).squeeze(-1)

        # Step 2: Sentence Embedding Generation
        #print("step 2")
        
        #semantic_query = self.semantic_query(torch.arange(self.semantic_query_num).to(device))
        #x = semantic_query.repeat(batch_size, 1, 1)
        #print(x.size())
        #print(pos_embeds.size())
        #print((vis_embeds+pos_embeds).size())
        #for i, blk in enumerate(self.transformer.blocks_topic):
            #x, _attn = blk(x, (vis_embeds+pos_embeds), (vis_embeds+pos_embeds))  
            #x, _attn = blk(x, (vis_embeds+pos_embeds), (vis_embeds))
            #x, _attn = blk(x, (vis_embeds), (vis_embeds))
        
        #x = self.transformer.norm_sent(x)
        #sent_feats = self.topic_proj(x).float()
        
        #x = self.transformer.norm_sent(x)
        #sent_feats = self.topic_proj(x).float()
        
        #print(sent_feats.shape)
        #print(vis_embeds[:,0,:].unsqueeze(1).repeat(1,self.semantic_query_num,1).shape)
        #topic_preds = self.topic_clas(torch.cat([sent_feats, vis_embeds[:,0,:].unsqueeze(1).repeat(1,self.semantic_query_num,1)], dim=2)).squeeze(-1)
        
        # Step 3: assignment of sent_feats prediction
        #topic_preds = self.topic_clas(vis_embeds[:,0,:])        #(bs, 200) 

        #classification_query = self.classification_query(torch.arange(self.semantic_query_num).to(device))
        #x = classification_query.repeat(batch_size, 1, 1)
        #x = sent_feats.clone()
        """
        for i, blk in enumerate(self.transformer.blocks_cls):
            #x, _attn = blk(x, (vis_embeds+pos_embeds+type_embeds)[:,1:], (vis_embeds+pos_embeds+type_embeds)[:,1:])  
            x, _attn = blk(x)  

        x = self.transformer.norm_topic(x)
        """
        #topic_preds = self.topic_clas2(x).squeeze(-1)
        

        #print(topic_preds.shape)
        
        match_idxs = self.matcher(sent_feats, topic_preds, sent_embeds, sent_num)
        #print(match_idxs)
        
        indices = self._get_src_permutation_idx(match_idxs)
        #include the most similar queries
        if phase=="train":
            expand_indices = self.expand_indices(sent_feats, indices, batch_size)
        else:
            expand_indices = indices

        #print(indices[0].shape)
        if phase == "train":
            # count positive and negative frequents
            neg_count = np.ones(self.semantic_query_num)*batch_size
            pos_count = np.zeros(self.semantic_query_num)
            for i in range(self.semantic_query_num):
                pos_count[i] += torch.sum(indices[1]==i).cpu().numpy()

            neg_count=neg_count-pos_count    
            self.train_select_pos_count = self.train_select_pos_count+pos_count
            self.train_select_neg_count = self.train_select_neg_count+neg_count
            

        ret_result=None
        pred_indices = None
        if phase!="train":  
            pred_indices = (topic_preds.sigmoid()>0.45).nonzero()
            
            top_k = 5
            top_indices = topic_preds.sigmoid().topk(top_k, dim=1).indices.view(-1)
            top_batches = torch.Tensor(np.arange(batch_size).repeat(top_k)).to(device)
            #pred_indices = (pred_indices[:,0], pred_indices[:,1]) 
            pred_indices = (torch.cat([pred_indices[:,0], top_batches], dim=0), torch.cat([pred_indices[:,1], top_indices], dim=0))
            
            indices_max = topic_preds.argmax(dim=1)
            
            #pred_indices = self.select_indices(sent_feats, batch_size, device)           
            pred_indices = self.select_indices_v2(sent_feats, pred_indices, indices_max, batch_size, device)

            neg_count = np.ones(self.semantic_query_num)*batch_size
            pos_count = np.zeros(self.semantic_query_num)
            for i in range(self.semantic_query_num):
                pos_count[i] += torch.sum(pred_indices[1]==i).cpu().numpy()
                
            neg_count=neg_count-pos_count
            self.test_select_pos_count = self.test_select_pos_count+pos_count
            self.test_select_neg_count = self.test_select_neg_count+neg_count
            
            pos_count = np.zeros(self.semantic_query_num)
            for i in range(self.semantic_query_num):
                pos_count[i] += torch.sum(indices[1]==i).cpu().numpy()
                
            self.test_select_count = self.test_select_count+pos_count

            ret_result = self.generate_ret_result(sent_feats, text_ids, pred_indices, indices, topic_preds, batch_size, device)

        ret = {
            "ids": image_ids,
            "path": image_path,
            "indices": indices,
            "expand_indices": expand_indices,
            "sent_feats": sent_feats,                #预测的sentence feature
            "sent_embeds": sent_embeds,              #ground-truth sentence embedding
            #"sent_masks": sent_masks[:,1:],         #mask of sentence sequence

            #"text_ids": text_ids[:,:,1:],           #List, 每句话的text ids
            #"text_masks": text_masks[:,:,1:],       #List, 每句话的mask
            #"text_feats": text_feats,               #List, 每句话的feature
            #"text_logits": text_logits,             #List, 每句话的预测logits
            
            "topic_preds": topic_preds,
            #"topic_label": topic_label,
            "ret_result" : ret_result,
            "attn": _attn,
            "patch_index": patch_index,
            "sent_num"   : sent_num,                 #batch中每个report的句子数
        }
        
        return ret

    def forward(self, batch):
        ret = dict()
        #print(self.current_tasks)
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        if "mimic" in self.current_tasks:
            ret_t, ret_result = objectives.compute_mimic(self, batch)
            ret.update(ret_t) 
            #print("forward", ret_result[0])  

        if "iuxray" in self.current_tasks:
            ret_t, ret_result = objectives.compute_iuxray(self, batch)
            ret.update(ret_t) 
            #print("forward", ret_result[0])          

        return ret, ret_result

    def update_topic_path(self, topic):
        start = 0
        #print(topic)
        for i in topic:
            self.path_graph[start, i+1]+=1                          #path_graph[0]:start; path_graph[101]:end; path_graph[i+1]: i-th topic
            start = i+1
        self.path_graph[start, self.semantic_query_num+1]+=1
        #print(self.path_graph.nonzero())

    def save_ret_result(self, ret_result, ids=None, path=None, attn=None, patch=None):
        #print(path)
        ret_result_zip = zip(ret_result[0], ret_result[1], ret_result[2], ret_result[3], ret_result[4])
        for idx, (sent_ret, sent_tar, s, tp, bleu) in enumerate(ret_result_zip):
            self.ret_result_file.write("====\n")
            self.ret_result_file.write("pred: {}\n".format(sent_ret))
            self.ret_result_file.write("targ: {}\n".format(sent_tar))
            self.ret_result_file.write("pred prob: {}\n".format(tp[0]))
            self.ret_result_file.write("gt prob: {}\n".format(tp[1]))
            #self.ret_result_file.write("gt prob: {}\n".format(tp))
            self.ret_result_file.write("retrieval sim: {}\n".format(s))
            self.ret_result_file.write("bleu_1:{} bleu_2:{} bleu_3:{} bleu_4:{}\n".format(bleu[0], bleu[1], bleu[2], bleu[3]))
            self.update_topic_path(tp[1][1])

            if ids!=None:
                t_dict = dict()
                #print(path[0][idx])
                t_dict["path"]      = path[0][idx]
                t_dict["pred"]      = sent_ret
                t_dict["targ"]      = sent_tar
                t_dict["pred_prob"] = tp[0][0].tolist()
                t_dict["pred_topic"] = tp[0][1].tolist()
                t_dict["gt_prob"]   = tp[1][0].tolist()
                t_dict["gt_topic"] = tp[1][1].tolist()
                #t_dict["attn"]      = attn[idx].tolist()
                #t_dict["patch"]     = patch[0][idx].tolist()          
                t_dict["ret_sim"]   = s
                t_dict["bleu"]      = [float(bleu[0]), float(bleu[1]), float(bleu[2]), float(bleu[3])] 
                self.test_log[ids[idx]] = t_dict
                print(t_dict)

    def training_step(self, batch, batch_idx):
        mrg_utils.set_task(self)
        output, ret_result = self(batch)
        #print(output.items())
        #total_loss = sum([v for k, v in output.items() if "loss" in k])
        total_loss = output
        #print("total_loss", total_loss)
        return total_loss

    def training_epoch_end(self, outs):
        print("training_epoch_end")
        mrg_utils.epoch_wrapup(self)

        #print(self.train_select_pos_count)
        #print(self.train_select_neg_count)
        
        class_freq = dict()
        class_freq["class_freq"] = self.train_select_pos_count
        class_freq["neg_class_freq"] = self.train_select_neg_count
        mmcv.dump(class_freq, "iuxray_class_freq.pkl")
        mmcv.dump(self.path_graph, "./path_graph.pkl")
        print("iuxray_class_freq.pkl updated.")
        print("path_graph.pkl updated.")
        self.train_select_pos_count = np.zeros(self.semantic_query_num)
        self.train_select_neg_count = np.zeros(self.semantic_query_num)
        self.path_graph = np.zeros((self.semantic_query_num+2, self.semantic_query_num+2))

    def validation_step(self, batch, batch_idx):
        #print("validation_step")
        mrg_utils.set_task(self)
        output, ret_result = self(batch)
        #print(ret_result)
        self.save_ret_result(ret_result)

    def validation_epoch_end(self, outs):
        #print("validation_epoch_end")
        mrg_utils.epoch_wrapup(self)
        self.ret_result_file.close()
        self.epoch_count+=1
        self.ret_result_file = open("./ret_logs/ret_{}.txt".format(self.epoch_count), "w")
        
        #print(self.test_select_pos_count)
        #print(self.test_select_neg_count)
        #print(self.test_select_count)
        
        self.test_select_pos_count = np.zeros(self.semantic_query_num)
        self.test_select_neg_count = np.zeros(self.semantic_query_num)  
        self.test_select_count = np.zeros(self.semantic_query_num)   



    def test_step(self, batch, batch_idx):
        #print("test_step")
        mrg_utils.set_task(self)
        output, ret_result = self(batch)
        ids = output["ids"]
        path = output["path"]
        #print(path)
        attn = output["attn"]
        patch = output["patch"]
        self.save_ret_result(ret_result, ids, path, attn, patch)
        #ret = dict()

        #if self.hparams.config["loss_names"]["vqa"] > 0:
        #    ret.update(objectives.vqa_test_step(self, batch, output))
        return output#ret

    def test_epoch_end(self, outs):
        #model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        #if self.hparams.config["loss_names"]["vqa"] > 0:
        #    objectives.vqa_test_wrapup(outs, model_name)
        mrg_utils.epoch_wrapup(self)
        self.ret_result_file.close()
        mmcv.dump(self.path_graph, "./path_graph.pkl")
        mmcv.dump(self.test_log, "./ret_logs/test_log.pkl")
        with open("./ret_logs/test_log.json", 'w') as f:
            json.dump(self.test_log, f)

    def configure_optimizers(self):
        return mrg_utils.set_schedule(self)

if __name__=="main":
    model = TransformerSQ()