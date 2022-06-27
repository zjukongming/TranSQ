from sacred import Experiment

ex = Experiment("TranSQ")


def _loss_names(d):
    ret = {
        "itm": 0,
        "mlm": 0,
        "mpp": 0,
        "vqa": 0,
        "nlvr2": 0,
        "irtr": 0,
        "mimic": 0,
    }

    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "vilt"
    seed = 0
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Image setting
    train_transform_keys = ["pixelbert"]
    val_transform_keys = ["pixelbert"]
    image_size = 384
    max_image_len = 150
    patch_size = 32
    draw_false_image = 1
    image_only = False

    # Text Setting
    vqav2_label_size = 3129
    max_text_len = 80
    max_sent_num = 25
    sent_emb = 768
    tokenizer = "bert-base-uncased"
    semantic_query_num = 50
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_text = 0

    # Transformer Setting
    vit = "vit_base_patch32_384"
    hidden_size = 768
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-2
    #weight_decay = 0.01
    weight_decay = 1e-5
    decay_power = 1
    max_epoch = 100
    max_steps = 25000
    warmup_steps = 2500
    end_lr = 1e-7
    lr_mult = 1  # multiply lr for downstream heads

    # Downstream Setting
    get_recall_metric = False

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = True

    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1
    pre_data_path = ""
    #load_path = "./result/100_0_0.05/checkpoint.ckpt"
    #load_path = "./result/DBLoss_V18/checkpoint.ckpt"
    #load_path = "./result/DBLoss_v20/checkpoint.ckpt"
    load_path = "./pretrained-models/mimic.ckpt"
    #load_path = "./result/train_mimic_randaug_seed0_from_/version_21/checkpoints/last-5.ckpt"
    
    #load_path = ""

    num_workers = 2
    precision = 16


# Named configs for "environment" which define gpus and nodes, and paths@ex.named_config
def task_train_mimic():
    exp_name = "train_mimic_randaug"
    datasets = ["mimic"]
    train_transform_keys = ["pixelbert_randaug"]
    #loss_names = _loss_names({"mimic": 1})
    loss_names = _loss_names({"mimic": 1})
    batch_size = 64
    max_image_len = 200
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.02
    get_recall_metric = False
    draw_false_text = 15
    learning_rate = 1e-4
    data_root = "/mnt/vdb1/Data/mimic_cxr"
    pre_data_path = "/mnt/vdb1/Code/Medical-Report-Generation/TranSQ/preprocess/data"
    num_gpus = 1
    per_gpu_batchsize = 64
    val_check_interval = 1.0

# Named configs for "etc" which are orthogonal to "env" and "task", need to be added at the end


@ex.named_config
def step25k():
    max_epoch = 100
    max_steps = 25000


@ex.named_config
def step50k():
    max_epoch = 100
    max_steps = 50000


@ex.named_config
def step100k():
    max_epoch = 100
    max_steps = 100000


@ex.named_config
def step200k():
    max_epoch = 200
    max_steps = 200000


@ex.named_config
def vit32_base():
    vit = "vit_base_patch32_384"
    patch_size = 32
    hidden_size = 768
    num_heads = 12
    num_layers = 12
