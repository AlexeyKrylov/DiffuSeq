import argparse
import torch
import json, os
import time

from diffuseq import gaussian_diffusion as gd
from diffuseq.gaussian_diffusion import SpacedDiffusion, space_timesteps
from diffuseq.transformer_model import TransformerNetModel
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import pickle

class myTokenizer():
    """
    Load tokenizer from bert config or defined BPE vocab dict
    """
    ################################################
    ### You can custome your own tokenizer here. ###
    ################################################
    def __init__(self, args):
        # self.vocab_size = 7000
        # try:
        #     if args.test:
        #         with open("checkpoint-path/tokenizer.pkl", 'rb') as f:
        #             self.tokenizer = pickle.load(f)
        #             self.vocab_size = len(self.tokenizer.get_vocab())
        #             args.vocab_size = self.vocab_size  # update vocab size in args
        #             self.sep_token_id = 2
        #             self.pad_token_id = 3
        #         print("successful loading !!")
        #         return
        # except:
        #     print("TRAIN PHASE!")
        if args.vocab == 'bert':
            print(args.config_name)
            tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
            # with open('./datasets/SparQL/src-english_train_split.txt', 'r', encoding='utf8') as f:
            #     src = f.readlines()
            #
            # tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            # trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]"], vocab_size=3602)
            # tokenizer.pre_tokenizer = Whitespace()
            # tokenizer.train_from_iterator([i.strip() for i in src], trainer)
            print(len(tokenizer.get_vocab()))

            self.tokenizer = tokenizer

            with open('./datasets/SparQL/spec-english_train_split.txt', 'r') as f:
                data = f.read()

            self.tokenizer.add_tokens(data.replace('\n', ' ').split(' '))
            self.sep_token_id = 2
            self.pad_token_id = 3
            # save
            # with open("checkpoint-path/tokenizer.pkl", "wb") as f:
            #     pickle.dump(self.tokenizer, f)
        else:
            # load vocab from the path
            print('#'*30, 'load vocab from', args.vocab)
            vocab_dict = {'[START]': 0, '[END]': 1, '[UNK]':2, '[PAD]':3}
            with open(args.vocab, 'r', encoding='utf-8') as f:
                for row in f:
                    vocab_dict[row.strip().split(' ')[0]] = len(vocab_dict)
            self.tokenizer = vocab_dict
            self.rev_tokenizer = {v: k for k, v in vocab_dict.items()}
            self.sep_token_id = vocab_dict['[END]']
            self.pad_token_id = vocab_dict['[PAD]']
            # save
            if int(os.environ['LOCAL_RANK']) == 0:
                path_save_vocab = f'{args.checkpoint_path}/vocab.json'
                with open(path_save_vocab, 'w') as f:
                    json.dump(vocab_dict, f)
                
        self.vocab_size = len(self.tokenizer.get_vocab())
        args.vocab_size = self.vocab_size # update vocab size in args
    
    def encode_token(self, sentences):
        if isinstance(self.tokenizer, dict):
            input_ids = [[0] + [self.tokenizer.get(x, self.tokenizer['[UNK]']) for x in seq.split()] + [1] for seq in sentences]
        else:
            # input_ids = [i.ids for i in self.tokenizer.encode_batch(sentences, add_special_tokens=True)]
            input_ids = self.tokenizer(sentences, add_special_tokens=True)['input_ids']
        return input_ids
        
    def decode_token(self, seq):
        if isinstance(self.tokenizer, dict):
            seq = seq.squeeze(-1).tolist()
            while len(seq)>0 and seq[-1] == self.pad_token_id:
                seq.pop()
            tokens = " ".join([self.rev_tokenizer[x] for x in seq]).replace('__ ', '').replace('@@ ', '')
        else:
            seq = seq.squeeze(-1).tolist()
            while len(seq)>0 and seq[-1] == self.pad_token_id:
                seq.pop()
            tokens = self.tokenizer.decode(seq)
        return tokens


def load_model_emb(args, tokenizer):
    ### random emb or pre-defined embedding like glove embedding. You can custome your own init here.
    model = torch.nn.Embedding(tokenizer.vocab_size, args.hidden_dim)
    path_save = '{}/random_emb.torch'.format(args.checkpoint_path)
    path_save_ind = path_save + ".done"
    if int(os.environ['LOCAL_RANK']) == 0:
        if os.path.exists(path_save):
            print('reload the random embeddings', model)
            model.load_state_dict(torch.load(path_save))
        else:
            print('initializing the random embeddings', model)
            torch.nn.init.normal_(model.weight)
            if not os.path.exists('/'.join(path_save.split('/')[:-1])):
                os.mkdir('/'.join(path_save.split('/')[:-1]))
            torch.save(model.state_dict(), path_save)
            # os.sync() # ADD BY ME
            with open(path_save_ind, "x") as _:
                pass
    else:
        while not os.path.exists(path_save_ind):
            time.sleep(1)
        print('reload the random embeddings', model)
        model.load_state_dict(torch.load(path_save))

    return model, tokenizer


def load_tokenizer(args):
    tokenizer = myTokenizer(args)
    return tokenizer

def load_defaults_config():
    """
    Load defaults for training args.
    """
    with open('diffuseq/config.json', 'r') as f:
        return json.load(f)


def create_model_and_diffusion(
    hidden_t_dim,
    hidden_dim,
    vocab_size,
    config_name,
    use_plm_init,
    dropout,
    diffusion_steps,
    noise_schedule,
    learn_sigma,
    timestep_respacing,
    predict_xstart,
    rescale_timesteps,
    sigma_small,
    rescale_learned_sigmas,
    use_kl,
    notes,
    **kwargs,
):
    model = TransformerNetModel(
        input_dims=hidden_dim,
        output_dims=(hidden_dim if not learn_sigma else hidden_dim*2),
        hidden_t_dim=hidden_t_dim,
        dropout=dropout,
        config_name=config_name,
        vocab_size=vocab_size,
        init_pretrained=use_plm_init
    )

    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)

    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]

    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        rescale_timesteps=rescale_timesteps,
        predict_xstart=predict_xstart,
        learn_sigmas = learn_sigma,
        sigma_small = sigma_small,
        use_kl = use_kl,
        rescale_learned_sigmas=rescale_learned_sigmas
    )

    return model, diffusion


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
