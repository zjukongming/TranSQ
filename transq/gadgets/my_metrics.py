import torch
import torch.nn.functional as F
import pickle as pkl
from torch import Tensor, tensor
from sentence_transformers import SentenceTransformer
#from pytorch_lightning.metrics import Metric
from transq.pycocoevalcap.bleu.bleu import Bleu
from transq.pycocoevalcap.meteor import Meteor
from transq.pycocoevalcap.rouge import Rouge
from transq.pycocoevalcap.cider import Cider

from typing import Any, Callable, Optional, Sequence, Dict, List, Tuple, Union
#from transq.modules.gallery import Gallery

from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional.text.bleu import _bleu_score_compute, _bleu_score_update, _tokenize_fn
from torchmetrics.functional.text.rouge import ALLOWED_ROUGE_KEYS, _rouge_score_compute, _rouge_score_update
from torchmetrics.utilities.imports import _NLTK_AVAILABLE

import os 

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        #print("update")
        logits, target = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        preds = logits.argmax(dim=-1)
        preds = preds[target != -100]
        target = target[target != -100]
        if target.numel() == 0:
            return 1

        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total


class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar
        self.total += 1

    def compute(self):
        return self.scalar / self.total


class VQAScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().float().to(self.score.device),
            target.detach().float().to(self.score.device),
        )
        logits = torch.max(logits, 1)[1]
        one_hots = torch.zeros(*target.size()).to(target)
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = one_hots * target

        self.score += scores.sum()
        self.total += len(logits)

    def compute(self):
        return self.score / self.total

class MSEScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("mse_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, mse):       
        self.mse_score+=mse
        self.total+=1     

    def compute(self):
        return -self.mse_score / self.total

class MRGScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        print("MRG_init")
        self.score = {"BLEU_1":0, "BLEU_2":0, "BLEU_3":0, "BLEU_4":0, "METEOR":0, "ROUGE_L":0}
        self.total = 0
    def compute_scores(self, gts, res):
        """
        Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

        :param gts: Dictionary with the image ids and their gold captions,
        :param res: Dictionary with the image ids ant their generated captions
        :print: Evaluation score (the mean of the scores of all the instances) for each measure
        """

        # Set up scorers
        scorers = [
            (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L")
        ]
        eval_res = {}
        #print("gts",gts)
        #print("res",res)
        # Compute score for each metric
        for scorer, method in scorers:
            try:
                score, scores = scorer.compute_score(gts, res, verbose=0)
            except TypeError:
                score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, m in zip(score, method):
                    eval_res[m] = sc
            else:
                eval_res[method] = score
        return eval_res

    def update(self, tokenizer, logits, targets):       
        # Compute score for each metric
        print("update")
        val_gts, val_res = [], []
        for i in range(len(logits)):    
            preds = torch.argmax(logits[i], dim=2).cpu().numpy()
            gts   = targets[:, i, :].int().cpu().numpy()
            logit = tokenizer.decode_batch(preds)
            target = tokenizer.decode_batch(gts)
            
                #print(logits[i])
            print("{} pred".format(i), logit[0])  
            print("{} target".format(i), target[0]) 
            val_res.extend(logit)
            val_gts.extend(target)
            
        val_met = self.compute_scores({i: [gt] for i, gt in enumerate(val_gts)},
                                    {i: [re] for i, re in enumerate(val_res)})  
        for i in val_met:
            self.score[i] += val_met[i] 
        self.total+=1
        #return self.eval_res      

    def compute(self):
        #print(self.eval_res)
        metric = "BLEU_4"
        print("val_metrics", self.total)
        for i in self.score:
            #print(i, self.score[i], self.total, self.score[i]/self.total)
            print(i, self.score[i]/self.total)
        return self.score[metric] / self.total

class MRG_Retrieval(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        #self.score = {"BLEU_1":0, "BLEU_2":0, "BLEU_3":0, "BLEU_4":0, "METEOR":0, "ROUGE_L":0}
        self.score = {"METEOR":0, "ROUGE_L":0}
        self.total = 0
        #f = open("sentence_gallery.pkl", "rb")

    def compute_scores(self, gts, res):
        """
        Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

        :param gts: Dictionary with the image ids and their gold captions,
        :param res: Dictionary with the image ids ant their generated captions
        :print: Evaluation score (the mean of the scores of all the instances) for each measure
        """

        # Set up scorers
        scorers = [
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
        ]
        eval_res = {}
        # Compute score for each metric
        for scorer, method in scorers:
            try:
                score, scores = scorer.compute_score(gts, res)
            except TypeError:
                score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, m in zip(score, method):
                    eval_res[m] = sc
            else:
                eval_res[method] = score
        return eval_res

    def update(self, vectors, targets):           
        #sent_ret, sim = self.gallery.check_gallery_cosine(vectors.cpu().numpy())
        #print(targets)
        #sent_targ = tokenizer.decode_batch(targets.int().cpu().numpy())
        #for ret, tar, s in zip(sent_ret, sent_targ, sim):               
        val_met = self.compute_scores({i: [gt] for i, gt in enumerate(targets)},
                                    {i: [re] for i, re in enumerate(vectors)})  
            
        for i in val_met:
            self.score[i] += val_met[i] 
        self.total+=1
    
    def compute(self):
        ret = dict()
        for i in self.score:
            ret[i] = self.score[i]/(self.total+1e-7)
            #print(i, self.score[i]/self.total)
        return ret

class TopicAccuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("pos_correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("pos_total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("neg_correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("neg_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, pred, target):
        #print("update")
        pred, target = (
            pred.detach().to(self.correct.device).squeeze(2),
            target.detach().to(self.correct.device),
        )
        pred = F.sigmoid(pred)>0.5
        #print("Positive:", torch.sum((pred==target)*target)/torch.sum(target))
        #print("Negative:", torch.sum((pred==target)*(1-target))/torch.sum(1-target))

        self.correct += torch.sum(pred == target)
        self.pos_correct += torch.sum((pred==target)*target)
        self.neg_correct += torch.sum((pred==target)*(1-target))
        self.total += target.numel()
        self.pos_total += torch.sum(target)
        self.neg_total += torch.sum(1-target)        

    def compute(self):
        print(self.pos_correct / self.pos_total)
        print(self.neg_correct / self.neg_total)
        return self.correct / self.total

class BLEUScore(Metric):
    """Calculate `BLEU score`_ of machine translated text with one or more references.
    Args:
        n_gram:
            Gram value ranged from 1 to 4 (Default 4)
        smooth:
            Whether or not to apply smoothing – see [2]
        compute_on_step:
            Forward only calls ``update()`` and returns None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step.
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When `None`, DDP
            will be used to perform the allgather.
    Example:
        >>> translate_corpus = ['the cat is on the mat'.split()]
        >>> reference_corpus = [['there is a cat on the mat'.split(), 'a cat is on the mat'.split()]]
        >>> metric = BLEUScore()
        >>> metric(reference_corpus, translate_corpus)
        tensor(0.7598)
    References:
        [1] BLEU: a Method for Automatic Evaluation of Machine Translation by Papineni,
        Kishore, Salim Roukos, Todd Ward, and Wei-Jing Zhu `BLEU`_
        [2] Automatic Evaluation of Machine Translation Quality Using Longest Common Subsequence
        and Skip-Bigram Statistics by Chin-Yew Lin and Franz Josef Och `Machine Translation Evolution`_
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update= True

    trans_len: Tensor
    ref_len: Tensor
    numerator: Tensor
    denominator: Tensor

    def __init__(
        self,
        n_gram: int = 4,
        smooth: bool = False,
        weights: Optional[Sequence[float]] = None,
        #compute_on_step: bool = True,
        #dist_sync_on_step: bool = False,
        #process_group: Optional[Any] = None,
        #dist_sync_fn: Optional[Callable] = None,
    ):
        super().__init__(
            #compute_on_step=compute_on_step,
            #dist_sync_on_step=dist_sync_on_step,
            #process_group=process_group,
            #dist_sync_fn=dist_sync_fn,
        )

        self.n_gram = n_gram
        self.smooth = smooth
        self.weights = weights if weights is not None else [1.0 / n_gram] * n_gram

        self.add_state("trans_len", tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("ref_len", tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("numerator", torch.zeros(self.n_gram), dist_reduce_fx="sum")
        self.add_state("denominator", torch.zeros(self.n_gram), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("add_all", default=tensor(0, dtype=torch.float), dist_reduce_fx="sum")

    def update(  # type: ignore
        self, preds: Sequence[str], target: Sequence[Sequence[str]]
    ) -> None:
        """Compute Precision Scores.
        Args:
            preds: An iterable of machine translated corpus
            target: An iterable of iterables of reference corpus
        """
        

        self.trans_len, self.ref_len = _bleu_score_update(
            preds,
            target,
            self.numerator,
            self.denominator,
            self.trans_len,
            self.ref_len,
            self.n_gram,
            _tokenize_fn,
        )

        curr_score = _bleu_score_compute(
            self.trans_len, self.ref_len, self.numerator, self.denominator, self.n_gram, self.weights,self.smooth
        )
        #print(trans_len)
        #print(ref_len)
        #print(curr_score)
        self.total+=1
        self.add_all+=curr_score
        #return curr_score

    def compute(self) -> Tensor:
        """Calculate BLEU score.
        Return:
            Tensor with BLEU Score
        """
        #return _bleu_score_compute(
        #    self.trans_len, self.ref_len, self.numerator, self.denominator, self.n_gram, 1,self.smooth
        #)
        #print("BLEU {} compute".format(self.n_gram),self.add_all, self.total)
        return self.add_all/self.total

class ROUGEScore(Metric):
    """`Calculate Rouge Score`_, used for automatic summarization. This implementation should imitate the behaviour
    of the `rouge-score` package `Python ROUGE Implementation`
    Args:
        use_stemmer:
            Use Porter stemmer to strip word suffixes to improve matching.
        rouge_keys:
            A list of rouge types to calculate.
            Keys that are allowed are ``rougeL``, ``rougeLsum``, and ``rouge1`` through ``rouge9``.
        compute_on_step:
            Forward only calls ``update()`` and returns None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step.
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When `None`, DDP
            will be used to perform the allgather.
    Example:
        >>> targets = "Is your name John"
        >>> preds = "My name is John"
        >>> rouge = ROUGEScore()   # doctest: +SKIP
        >>> from pprint import pprint
        >>> pprint(rouge(preds, targets))  # doctest: +NORMALIZE_WHITESPACE +SKIP
        {'rouge1_fmeasure': 0.25,
         'rouge1_precision': 0.25,
         'rouge1_recall': 0.25,
         'rouge2_fmeasure': 0.0,
         'rouge2_precision': 0.0,
         'rouge2_recall': 0.0,
         'rougeL_fmeasure': 0.25,
         'rougeL_precision': 0.25,
         'rougeL_recall': 0.25,
         'rougeLsum_fmeasure': 0.25,
         'rougeLsum_precision': 0.25,
         'rougeLsum_recall': 0.25}
    Raises:
        ValueError:
            If the python packages ``nltk`` is not installed.
        ValueError:
            If any of the ``rouge_keys`` does not belong to the allowed set of keys.
    References:
        [1] ROUGE: A Package for Automatic Evaluation of Summaries by Chin-Yew Lin `Rouge Detail`_
    """

    higher_is_better = True

    def __init__(
        self,
        use_stemmer: bool = False,
        rouge_keys: Union[str, Tuple[str, ...]] = ("rouge1", "rouge2", "rougeL", "rougeLsum"),  # type: ignore
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable] = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        if use_stemmer or "rougeLsum" in rouge_keys:
            if not _NLTK_AVAILABLE:
                raise ValueError("Stemmer and/or `rougeLsum` requires that nltk is installed. Use `pip install nltk`.")
            import nltk

        if not isinstance(rouge_keys, tuple):
            rouge_keys = tuple([rouge_keys])
        for key in rouge_keys:
            if key not in ALLOWED_ROUGE_KEYS:
                raise ValueError(f"Got unknown rouge key {key}. Expected to be one of {ALLOWED_ROUGE_KEYS}")

        self.rouge_keys = rouge_keys
        self.rouge_keys_values = [ALLOWED_ROUGE_KEYS[key] for key in rouge_keys]
        self.stemmer = nltk.stem.porter.PorterStemmer() if use_stemmer else None

        # Adding stated dynamically to prevent IndexError during sync function as some lists can be empty.
        for rouge_key in self.rouge_keys:
            for score in ["fmeasure", "precision", "recall"]:
                self.add_state(f"{rouge_key}_{score}", [], dist_reduce_fx=None)

    def update(self, preds: Union[str, List[str]], targets: Union[str, List[str]]) -> None:  # type: ignore
        """Compute rouge scores.
        Args:
            preds: An iterable of predicted sentences or a single predicted sentence.
            targets: An iterable of target sentences or a single target sentence.
        """

        if isinstance(preds, str):
            preds = [preds]

        if isinstance(targets, str):
            targets = [targets]

        output: Dict[Union[int, str], List[Dict[str, Tensor]]] = _rouge_score_update(
            preds, targets, self.rouge_keys_values, stemmer=self.stemmer
        )
        for rouge_key, metrics in output.items():
            for metric in metrics:
                for type, value in metric.items():
                    getattr(self, f"rouge{rouge_key}_{type}").append(value.to(self.device))

    def compute(self) -> Dict[str, Tensor]:
        """Calculate (Aggregate and provide confidence intervals) ROUGE score.
        Return:
            Python dictionary of rouge scores for each input rouge key.
        """
        update_output = {}
        for rouge_key in self.rouge_keys_values:
            for type in ["fmeasure", "precision", "recall"]:
                update_output[f"rouge{rouge_key}_{type}"] = getattr(self, f"rouge{rouge_key}_{type}")

        return _rouge_score_compute(update_output)

    def __hash__(self) -> int:
        # override to hash list objects.
        # this is a bug in the upstream pytorch release.
        hash_vals = [self.__class__.__name__]

        for key in self._defaults:
            value = getattr(self, key)
            if isinstance(value, list):
                value = tuple(value)
            hash_vals.append(value)

        return hash(tuple(hash_vals))        