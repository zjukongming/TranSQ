'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
from transq.datamodules.tokenizer import Tokenizer
data_dir = "/big-disk/mimic_cxr/"
threshold = 3
data_name = "mimic"
tokenizer = Tokenizer(data_dir, threshold, data_name)
class GPT2Config(object):
    def __init__(
            self,
            vocab_size_or_config_json_file=tokenizer.get_vocab_size(),
            n_positions=1024,
            n_ctx=1024,
            n_embd=768,
            n_layer=6,
            n_head=6,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            num_slots =3,
    ):
        self.vocab_size = vocab_size_or_config_json_file
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.num_slots = num_slots