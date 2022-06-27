from sentence_transformers import SentenceTransformer
from sklearn.metrics import mean_squared_error, r2_score
import os
import json
import numpy as np
import pickle as pkl
import torch
import torch.nn.functional as F

class Gallery():
    def __init__(self, filepath=None, split="train"):
        self.path=""
        self.ann = {}
        self.examples = []
        self.sent_encoder = None
        self.sentence_gallery = []
        self.sentence_vectors = []
        if filepath!= None:
            self.load_from_file(filepath, split)

    def load_from_file(self, filepath, split):
        self.path = os.path.join(filepath, "annotation.json")
        self.ann = json.loads(open(self.path, 'r').read())
        self.examples = self.ann[split]
        self.sent_encoder = SentenceTransformer('all-mpnet-base-v2').encode

        self.sentence_gallery = []

        for example_i in self.examples:
            sentences = example_i['report'].replace("\n"," ").split(". ") 
            for sent_i in sentences:
                self.sentence_gallery.append(sent_i) 
            #if len(self.sentence_gallery)>10000:
            #    break
        print(len(self.sentence_gallery))
        self.sentence_gallery = list(set(self.sentence_gallery))

        print("Generating sentence vectors...")
        self.sentence_vectors = np.array(self.sent_encoder(self.sentence_gallery))
        sent_vecs_norm = np.linalg.norm(x = self.sentence_vectors, ord=2, axis = 1, keepdims = True)
        self.sent_vecs_norm = sent_vecs_norm.clip(min=1e-7).reshape(-1,1)
        
        print(len(self.sentence_vectors))
        """
        print("Excluting sentence vectors...")
        include_set = np.array([True]*len(self.sentence_vectors))
        
        normed_sent_vecs = self.sentence_vectors/self.sent_vecs_norm
        for i in range(len(normed_sent_vecs)):
            if i%10000==0:
                print(i,"/",len(normed_sent_vecs))
            if include_set[i]==False:
                continue
            v_i = normed_sent_vecs[i]
            sim = v_i.dot(normed_sent_vecs.transpose(1,0))
            include_set[sim>0.95]=False
            include_set[i]=True
        print(include_set.sum())
        self.sentence_gallery = np.array(self.sentence_gallery)[include_set]
        self.sentence_vectors = self.sentence_vectors[include_set]
        self.sent_vecs_norm = self.sent_vecs_norm[include_set]
        """
        #self.normed_sentence_vectors = self.sentence_vectors/sent_vecs_norm

    def check_gallery(self, vec):
        mse = []
        vec_num, vec_dim = self.sentence_vectors.shape
        mse = np.sum((self.sentence_vectors-vec)**2, axis=1)/vec_dim
        idx = np.argmin(mse)
        
        return self.sentence_gallery[idx], mse[idx]

    def check_gallery_cosine(self, vec):
        vec_norm = np.linalg.norm(x = vec, ord=2, axis = 1, keepdims = True)
        vec_norm = vec_norm.clip(min=1e-7).reshape(-1,1)
        normed_vec = vec/vec_norm
        dist = normed_vec.dot((self.sentence_vectors/self.sent_vecs_norm).transpose(1,0))
        idx = np.argmax(dist, axis=1)
        dists = []
        sents = []
        print(type(idx))
        for i in range(len(idx)):
            dists.append(dist[i][idx[i]])
            sents.append(self.sentence_gallery[idx[i]])
        return sents, dists

    def check_gallery_sim(self, vec, eps=1e-6):
        vecs_mean = self.sentence_vectors.mean(axis=1)[:,np.newaxis]
        vec_std  = np.sqrt(self.sentence_vectors.var(axis=1)+eps)[:, np.newaxis]
        norm_sents = (self.sentence_vectors-vecs_mean)/vec_std
        dist = vec.dot(norm_sents.transpose(1,0))
        idx = np.argmax(dist, axis=1)
        dists = []
        sents = []
        print(type(idx))
        for i in range(len(idx)):
            dists.append(dist[i][idx[i]])
            sents.append(self.sentence_gallery[idx[i]])
        return sents, dists        

    def __getitem__(self, index):
        sent = self.sentence_gallery[index]
        vec = self.sentence_vectors[index]
        return sent, vec

    def __len__(self):
        return(len(self.sentence_gallery))

#path = "/mnt/vdb1/Data/mimic_cxr"
#save
"""
sent_gallery = Gallery(path)
print(len(sent_gallery))
f = open("sentence_gallery_test.pkl", "wb")
pkl.dump(sent_gallery, f)
"""

#load
"""
path = "/mnt/vdb1/Data/mimic_cxr"
f = open("sentence_gallery.pkl", "rb")
sent_gallery = Gallery()
sent_gallery = pkl.load(f)
val_gallery = Gallery(path, "val")
for sent_i, vec_i in val_gallery:
    #sent_gallery.check_gallery(sent_vec*10+np.random.rand(768)/10)

    vec = vec_i[np.newaxis,:]
    print(sent_i)
    print(sent_gallery.check_gallery_cosine(vec))
"""