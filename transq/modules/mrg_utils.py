import torch
import random

from transformers.optimization import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from transq.modules.dist_utils import all_gather
#from transq.modules.objectives import compute_irtr_recall
from transq.gadgets.my_metrics import Accuracy, VQAScore, Scalar, MRGScore, MRG_Retrieval, MSEScore, TopicAccuracy, BLEUScore


def set_metrics(pl_module):
    #print(pl_module.hparams.config["loss_names"])
    for split in ["train", "val"]:
        for k, v in pl_module.hparams.config["loss_names"].items():
            if v < 1:
                continue

            if k == "mimic":
                #print(set_metrics, f"{split}_{k}_score")
                #setattr(pl_module, f"{split}_{k}_score", MRGScore())
                
                setattr(pl_module, f"{split}_{k}_bleu1", BLEUScore(1))
                setattr(pl_module, f"{split}_{k}_bleu2", BLEUScore(2))
                setattr(pl_module, f"{split}_{k}_bleu3", BLEUScore(3))
                setattr(pl_module, f"{split}_{k}_bleu4", BLEUScore(4))
                setattr(pl_module, f"{split}_{k}_mrg_score", MRG_Retrieval())
                setattr(pl_module, f"{split}_{k}_score", MSEScore())
                #setattr(pl_module, f"{split}_{k}_score", TopicAccuracy())
            else:
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
    #print(pl_module.train_itm_accuracy)

def epoch_wrapup(pl_module):

    print("epoch_wrapup")
    phase = "train" if pl_module.training else "val"
    the_metric = 0

    if pl_module.hparams.config["get_recall_metric"] and not pl_module.training:
        (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10) = compute_irtr_recall(pl_module)
        print((ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10), pl_module.global_step)
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r1", ir_r1, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r5", ir_r5, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r10", ir_r10, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r1", tr_r1, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r5", tr_r5, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r10", tr_r10, pl_module.global_step
        )
        the_metric += ir_r1.item() + tr_r1.item()

    for loss_name, v in pl_module.hparams.config["loss_names"].items():
        if v < 1:
            continue

        value = 0

        if loss_name == "mimic":
            #print("epoch_wrapup", f"{phase}_{loss_name}_score")
            print(f"{phase}_{loss_name}_bleu1", getattr(pl_module, f"{phase}_{loss_name}_bleu1").compute())
            print(f"{phase}_{loss_name}_bleu2", getattr(pl_module, f"{phase}_{loss_name}_bleu2").compute())
            print(f"{phase}_{loss_name}_bleu3", getattr(pl_module, f"{phase}_{loss_name}_bleu3").compute())
            print(f"{phase}_{loss_name}_bleu4", getattr(pl_module, f"{phase}_{loss_name}_bleu4").compute())
            print(f"{phase}_{loss_name}_mrg_score", getattr(pl_module, f"{phase}_{loss_name}_mrg_score").compute())
            value = getattr(pl_module, f"{phase}_{loss_name}_bleu4").compute()
            pl_module.log(f"{loss_name}/{phase}/score_epoch", value)
            print(f"{phase}_{loss_name}_score", value)
            getattr(pl_module, f"{phase}_{loss_name}_bleu1").reset()
            getattr(pl_module, f"{phase}_{loss_name}_bleu2").reset()
            getattr(pl_module, f"{phase}_{loss_name}_bleu3").reset()
            getattr(pl_module, f"{phase}_{loss_name}_bleu4").reset()
            getattr(pl_module, f"{phase}_{loss_name}_mrg_score").reset()
            getattr(pl_module, f"{phase}_{loss_name}_score").reset()
            #pl_module.log(
            #    f"{loss_name}/{phase}/loss_epoch",
            #    getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            #)
            #getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        else:
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

        the_metric += value
    pl_module.log(f"{phase}/the_metric", the_metric)
    #print(pl_module)
    print("epoch_wrapup", f"{phase}/the_metric", the_metric)
    return()


def check_non_acc_grad(pl_module):
    if pl_module.token_type_embeddings.weight.grad is None:
        return True
    else:
        grad = pl_module.token_type_embeddings.weight.grad
        return (grad.sum() == 0).item()


def set_task(pl_module):
    pl_module.current_tasks = [
        k for k, v in pl_module.hparams.config["loss_names"].items() if v >= 1
    ]
    #print("set_task", pl_module.current_tasks)
    return


def set_schedule(pl_module):
    lr = pl_module.hparams.config["learning_rate"]
    wd = pl_module.hparams.config["weight_decay"]

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    head_names = ["vqa_classifier", "nlvr2_classifier"]
    lr_mult = pl_module.hparams.config["lr_mult"]
    end_lr = pl_module.hparams.config["end_lr"]
    decay_power = pl_module.hparams.config["decay_power"]
    optim_type = pl_module.hparams.config["optim_type"]

    names = [n for n, p in pl_module.named_parameters()]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
            ],
            "weight_decay": wd,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and any(bb in n for bb in head_names)
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult,
        },
    ]

    if optim_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)

    if pl_module.trainer.max_steps is None:
        max_steps = (
            len(pl_module.trainer.datamodule.train_dataloader())
            * pl_module.trainer.max_epochs
            // pl_module.trainer.accumulate_grad_batches
        )
    else:
        max_steps = pl_module.trainer.max_steps

    warmup_steps = pl_module.hparams.config["warmup_steps"]
    if isinstance(pl_module.hparams.config["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    sched = {"scheduler": scheduler, "interval": "step"}

    return (
        [optimizer],
        [sched],
    )
