import argparse
import torch
import json, os
from diffuseq import gaussian_diffusion as gd
from diffuseq.gaussian_diffusion import SpacedDiffusion, space_timesteps
from diffuseq.transformer_model import TransformerNetModel
from transformers import AutoTokenizer
import pickle


class myTokenizer():
    """
    Load tokenizer from bert config or defined BPE vocab dict
    """
    ################################################
    ### You can custome your own tokenizer here. ###
    ################################################
    def __init__(self, args):
        self.args = args

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_config)

        if args.debug_mode:
            print(args.config_name)
            print("Number of tokens in tokenizer: ", len(tokenizer.get_vocab()))
            print(sorted(list(tokenizer.get_vocab().items()), key=lambda x : x[1])[:20])

        self.tokenizer = tokenizer

        list_of_add_toks = [
                            ' FROM',
                            ' AS',
                            'SELECT', 'select',
                            ' JOIN',
                            ' ON',
                            ' WHERE',
                            ' T1', ' t1',
                            ' T2', ' t2',
                            ' T3', ' t3',
                            ' T4', ' t4',
                            ' T5', ' t5',
                            ' GROUP BY',
                            ' ORDER BY',
                            ' AND',
                            ' DISTINCT',
                            ' DESC', ' desc',
                            ' LIMIT',
                            ' COUNT',
                            ' HAVING',
                            ' IN',
                            ' LIKE',
                            ' INTERSECT',
                            ' NOT',
                            ' OR',
                            ' ASC', ' asc',
                            ' MAX', ' max'
                            ' AVG', ' avg',
                            ' EXCEPT',
                            ' MIN', ' min',
                            ' BETWEEN',
                            ' UNION',
                            ' CAST',
                            ' SUM',
                            '[question]', ' ,',
                            ' [schema]', ' :',
                            ' [SEP_SCHEMS]'
                            ]

        if args.add_query_toks_path:
            with open(args.data_dir + args.add_query_toks_path, "r", encoding='utf8') as f:
                data = json.load(f)

            for i in data:
                list_of_add_toks.extend(i["query_toks"])

        self.tokenizer.add_tokens(list_of_add_toks)

        if args.debug_mode:
            print("Number of tokens in tokenizer after adding tokens: ", len(tokenizer.get_vocab()))

        if args.save_tokenizer:
            with open(args.checkpoint_path + "tokenizer.pkl", "wb") as f:
                pickle.dump(self.tokenizer, f)

        if args.vocab == 'bert':
            self.sep_token_id = 102
            self.pad_token_id = 0

        elif args.vocab == 't5':
            self.sep_token_id = 1
            self.pad_token_id = 0

        else:
            raise ValueError

        self.vocab_size = len(self.tokenizer.get_vocab())
        args.vocab_size = self.vocab_size # update vocab size in args
    
    def encode_token(self, sentences):
        input_ids = self.tokenizer(sentences, add_special_tokens=True)['input_ids']

        if self.args.debug_mode:
            print("All 4: ", [self.tokenizer.decode(x) for x in input_ids[:4]])
            print("First: ", [self.tokenizer.decode(x) for x in input_ids[0]])
            print("Second: ", [self.tokenizer.decode(x) for x in input_ids[1]])
            print("Third: ", [self.tokenizer.decode(x) for x in input_ids[2]])
            print("Fourth: ", [self.tokenizer.decode(x) for x in input_ids[3]])

        return input_ids
        
    def decode_token(self, seq):
        seq = seq.squeeze(-1).tolist()
        while len(seq) > 0 and seq[-1] == self.pad_token_id:
            seq.pop()

        if self.args.debug_mode:
            print("Decode sequence: ", [self.tokenizer.decode(x) for x in seq])

        return self.tokenizer.decode(seq)


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
    args,
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
):
    model = TransformerNetModel(
        args=args,
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
        args=args,
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
