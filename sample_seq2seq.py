"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json
from tqdm import tqdm
import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed
from diffuseq.rounding import denoised_fn_round, get_weights
from diffuseq.text_datasets import load_data_text
import time
from diffuseq.utils import dist_util, logger
from functools import partial
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_tokenizer
)


def create_argparser(sample='valid', path='./checkpoint-path/model0010000.pt'):
    defaults = dict(model_path=path, step=2000, out_dir='', top_p=1)
    decode_defaults = dict(split=sample, clamp_step=0, seed2=105, clip_denoised=False)
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main(filename):
    args = create_argparser('valid', path=f'./checkpoint-path/Bert/{filename}.pt').parse_args()

    dist_util.setup_dist()
    logger.configure()

    # load configurations.
    config_path = args.checkpoint_path + "training_args.json"
    print(config_path)

    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)

    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)

    logger.log("### Creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        args=args,
        hidden_t_dim=args.hidden_t_dim,
        hidden_dim=args.hidden_dim,
        vocab_size=args.vocab_size,
        config_name=args.config_name,
        use_plm_init=args.use_plm_init,
        dropout=args.dropout,
        diffusion_steps=args.diffusion_steps,
        noise_schedule=args.noise_schedule,
        learn_sigma=args.learn_sigma,
        timestep_respacing=args.timestep_respacing,
        predict_xstart=args.predict_xstart,
        rescale_timesteps=args.rescale_timesteps,
        sigma_small=args.sigma_small,
        rescale_learned_sigmas=args.rescale_learned_sigmas,
        use_kl=args.use_kl
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cuda:0")
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    model.to(dist_util.dev())
    model.eval()

    if args.use_fp16:
        model = model.half()

    tokenizer = load_tokenizer(args)
    model_emb = th.nn.Embedding(tokenizer.vocab_size, args.hidden_dim)
    model_emb.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())

    if args.use_fp16:
        model_emb = model_emb.half()

    model_emb_copy = get_weights(model_emb, args)

    set_seed(args.seed2)

    print("### Sampling...on", args.split)

    ## load data
    print(args.split)
    data_valid = load_data_text(
        batch_size=8,
        seq_len=args.seq_len,
        deterministic=True,
        data_args=args,
        split=args.split,
        loaded_vocab=tokenizer,
        loop=False
    )

    start_t = time.time()

    out_dir = os.path.join(args.out_dir, f"checkpoint-path")

    print("out_dir: ", out_dir)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_path = os.path.join(out_dir, f"{filename}.samples")
    print(out_path)
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    out_path = os.path.join(out_path, f"seed{args.seed2}_step{args.clamp_step}.json")
    print(out_path)

    all_test_data = []

    nofb = 0
    try:
        while nofb < 4:
        # while True:
            cond = next(data_valid)
            all_test_data.append(cond)
            nofb += 1
    except StopIteration:
        print('### End of reading iteration...')

    score = 0
    for cond in tqdm(all_test_data):

        input_ids_x = cond.pop('input_ids').to(dist_util.dev())
        x_start = model.get_embeds(input_ids_x)
        input_ids_mask = cond.pop('input_mask')
        attention_mask = cond.pop('attention_mask')
        input_ids_mask_ori = input_ids_mask

        noise = th.randn_like(x_start)
        input_ids_mask = th.broadcast_to(input_ids_mask.unsqueeze(dim=-1), x_start.shape).to(dist_util.dev())
        x_noised = th.where(input_ids_mask==0, x_start, noise)
        if args.use_fp16:
            x_noised = x_noised.half()

        model_kwargs = {}

        if args.step == args.diffusion_steps:
            args.use_ddim = False
            step_gap = 1
        else:
            args.use_ddim = True
            print(args)
            step_gap = args.diffusion_steps//args.step

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        sample_shape = (x_start.shape[0], args.seq_len, args.hidden_dim)

        samples = sample_fn(
            model,
            sample_shape,
            noise=x_noised,
            clip_denoised=args.clip_denoised,
            denoised_fn=partial(denoised_fn_round, args, model_emb_copy.cuda()),
            model_kwargs=model_kwargs,
            top_p=args.top_p,
            clamp_step=args.clamp_step,
            clamp_first=True,
            mask=input_ids_mask,
            x_start=x_start,
            gap=step_gap,
            attention_mask=attention_mask
        )

        model_emb_copy.cpu()
        # print(samples[0].shape) # samples for each step

        sample = samples[-1]
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        all_sentence = [sample.cpu().numpy() for sample in gathered_samples]

        word_lst_recover = []
        word_lst_ref = []
        word_lst_source = []


        arr = np.concatenate(all_sentence, axis=0)
        x_t = th.tensor(arr).cuda()


        reshaped_x_t = x_t.to(x_start.dtype)
        logits = model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab

        cands = th.topk(logits, k=1, dim=-1)


        for seq, input_mask in zip(cands.indices, input_ids_mask_ori):
            len_x = args.seq_len - sum(input_mask).tolist()
            tokens = tokenizer.decode_token(seq[len_x:])
            word_lst_recover.append(tokens)

        for seq, input_mask in zip(input_ids_x, input_ids_mask_ori):
            len_x = args.seq_len - sum(input_mask).tolist()
            word_lst_source.append(tokenizer.decode_token(seq[:len_x]))
            word_lst_ref.append(tokenizer.decode_token(seq[len_x:]))

        fout = open(out_path, 'a')
        for (recov, ref, src) in zip(word_lst_recover, word_lst_ref, word_lst_source):
            score += (recov == ref)
            print(json.dumps({"recover": recov, "reference": ref, "source": src}))
            print(json.dumps({"recover": recov, "reference": ref, "source": src}), file=fout)
        fout.close()
        print(score)

    print(score)
    print('### Total takes {:.2f}s .....'.format(time.time() - start_t))
    print(f'### Written the decoded output to {out_path}')


if __name__ == "__main__":
    main("model002000")
