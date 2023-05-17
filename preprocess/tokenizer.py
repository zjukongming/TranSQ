import json
import re
import os
from collections import Counter

class Tokenizer(object):
    def __init__(self, path, threshold, dataset_name):
        self.ann_path = os.path.join(path, "annotation.json")
        self.threshold = threshold
        self.dataset_name = dataset_name
        if self.dataset_name == 'iu_xray':
            self.clean_report = self.clean_report_iu_xray
        else:
            self.clean_report = self.clean_report_mimic_cxr
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.path = "/fast-disk/kongming/Code/TranSQ-github/preprocess/train_label_mimic.txt"
        self.txt = open(self.path,"r",errors='ignore').read()
        self.token2idx, self.idx2token = self.create_vocabulary()

    def create_vocabulary(self):
        txt = self.txt.lower()
        for ch in '!"#$&()*+,-./:;<=>?@[\\]^_{|}·~‘’':
            txt = txt.replace(ch,"")
        words = txt.split()

        counter = Counter(words)
        vocab = [k for k, v in counter.items()] + ['<unk>']+['<pad>']+['</s>']+['</e>']
        vocab.sort()
        #print(vocab)
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx
            idx2token[idx] = token
        #print(idx2token)
        return token2idx, idx2token

    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report)
        for ch in '!"#$&()*+,-./:;<=>?@[\\]^_{|}·~‘’':
            tokens = tokens.replace(ch,"")
        tokens = tokens.split()

        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids =ids
        #print("token_call", ids)
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.get_token_by_id(idx)
            else:
                break
        return txt


    def decode_txt(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                cur_txt = self.get_token_by_id(idx)
                
                if cur_txt=="<pad>" or cur_txt=="</s>" :
                    continue
                elif cur_txt=="</e>":
                    break
                
                txt = txt+cur_txt
                
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode_txt(ids))
        return out

