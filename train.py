"""
Train a diffusion model.
"""

import argparse
import json, os
from diffuseq.utils import dist_util, logger
from diffuseq.text_datasets import load_data_text
from diffuseq.step_sample import create_named_schedule_sampler
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    load_tokenizer
)
from train_util import TrainLoop
from transformers import set_seed
import wandb


def wandb_set(mode, key):
    ### custom your wandb setting here ###
    os.environ["WANDB_API_KEY"] = key
    os.environ["WANDB_MODE"] = mode
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def create_argparser():
    defaults = dict()
    defaults.update(load_defaults_config())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults) # update latest args according to argparse
    return parser

def main():
    args = create_argparser().parse_args() # get args from config.json

    if args.debug_mode:
        print(args)

    wandb_set(args.wandb_mode, args.wandb_api_key) # set wandb

    set_seed(args.seed) # set seed

    dist_util.setup_dist() # for Multi-GPU

    logger.configure() # Logger
    logger.log("### Creating data loader...")

    tokenizer = load_tokenizer(args) # load tokenizer from basic_utils.py

    if not args.one_batch:
        data = load_data_text(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            split='train',
            data_args=args,
            loaded_vocab=tokenizer,
            loop=True
        )

        data_valid = load_data_text(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            data_args=args,
            split='valid',
            deterministic=True,
            loaded_vocab=tokenizer,
        )
    else:
        data = load_data_text(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            data_args=args,
            split='valid',
            deterministic=True,
            loaded_vocab=tokenizer,
            nofb=1,
            nofs=args.batch_size
        )


        data_valid = load_data_text(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            data_args=args,
            split='valid',
            deterministic=True,
            loaded_vocab=tokenizer,
            nofb=1,
            nofs=args.batch_size
        )


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

    if args.resume_checkpoint:
        model.load_state_dict(
            dist_util.load_state_dict(args.resume_checkpoint, map_location="cpu")
        )

    model.to(dist_util.dev())

    if args.debug_mode:
        print("Model: ", model)

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    if args.debug_mode:
        print("Number of parameters: ", pytorch_total_params)

    logger.log(f'### The parameter count is {pytorch_total_params}')

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f'### Saving the hyperparameters to {args.checkpoint_path}/training_args.json')
    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if ('LOCAL_RANK' not in os.environ) or (int(os.environ['LOCAL_RANK']) == 0):
        wandb.init(
            project=os.getenv("WANDB_PROJECT", args.wandb_project_name),
            name=args.checkpoint_path,
        )
        wandb.config.update(args.__dict__, allow_val_change=True)

    logger.log("### Training...")

    TrainLoop(
        args=args,
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        resume_step=args.resume_step,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        learning_steps=args.learning_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=data_valid,
        eval_interval=args.eval_interval
    ).run_loop()

if __name__ == "__main__":
    main()
