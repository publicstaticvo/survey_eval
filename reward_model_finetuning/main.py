#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import math
import tqdm
import random
import numpy as np

os.environ['NCCL_DEBUG'] = "WARNING"
os.environ['NCCL_IB_DISABLE'] = "1"
os.environ['NCCL_P2P_DISABLE'] = "1"
os.environ['NCCL_SOCKET_IFNAME'] = "lo"

import torch
import torch.distributed as dist
from torch.nn.functional import logsigmoid
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import get_scheduler, AutoTokenizer, AutoModelForSequenceClassification, set_seed

import deepspeed
from deepspeed.ops.adam import FusedAdam
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import get_optimizer_grouped_parameters, set_random_seed, print_rank_0, save_model, barrier
from config import P2QConfig
from dataset import load_train_eval_data, P2QDataCollator


def print_rank(msg, rank=0):
    print(f"[Rank={rank}] {msg}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="rm.yaml")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def validate(model, tokenizer, config: P2QConfig, dataset: list[dict[str, str]], rank: int, device):
    if not dataset: return
    # eval_sampler = DistributedSampler(dataset) if rank >= 0 else SequentialSampler(dataset)
    # eval_dataloader = DataLoader(dataset, collate_fn=collator, sampler=eval_sampler, batch_size=config.per_device_eval_batch_size)
    model.eval()
    batch_size = config.per_device_eval_batch_size
    world_size = dist.get_world_size() if rank >= 0 else 1
    # scores[0] -- correct
    #     scores[0][0] -- seed > hard
    #     scores[0][1] -- seed > easy
    #     scores[0][2] -- hard > easy
    # scores[1] -- total
    # scores[2] -- scores
    scores = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    it = range(0, len(dataset), world_size * batch_size)
    for i in it:
        data = dataset[i:i + batch_size] if rank == -1 else dataset[i + rank:i + rank + world_size * batch_size:world_size]
        if not data: continue
        # 先处理数据
        batch, batch_id = [], []
        for x in data:
            for k, key in enumerate(['positive', 'hard_negative', 'easy_negative']):
                if x[key]:
                    batch.append(f"{x['topics']}[SEP]{x[key]['title']}. {x[key]['abstract']}")
                    batch_id.append(f"{x['id']}-{k}")        
        batch = tokenizer(batch, return_tensors='pt', padding='max_length', max_length=1024, truncation=True).to(device)
        batch = {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}
        with torch.no_grad(): output = model(**batch).logits
        output = output[:, 0].tolist()
        last_xid = None
        for o, j in zip(output, batch_id):
            xid, k = j.split("-")
            k = int(k)
            scores[2][k] += o 
            scores[3][k] += 1
            match k:
                case 0:
                    last_xid = [xid, o, None]
                case 1:
                    if last_xid[0] == xid and last_xid[1] is not None:
                        scores[1][0] += 1                     
                        if o < last_xid[1]: scores[0][0] += 1
                        last_xid[2] = o
                    else:
                        last_xid = [xid, None, o]
                case 2:
                    if last_xid[0] == xid:
                        if last_xid[1] is not None:
                            scores[1][1] += 1                     
                            if o < last_xid[1]: scores[0][1] += 1
                        if last_xid[2] is not None:
                            scores[1][2] += 1                     
                            if o < last_xid[2]: scores[0][2] += 1
    scores = torch.FloatTensor(scores).to(model.device)
    try:
        if rank >= 0: dist.all_reduce(scores)
        total_acc = (scores[0].sum() / scores[1].sum())
        scores[0] /= scores[1]
        scores[2] /= scores[3]
        scores = scores.tolist()
        print_rank_0(f"==============Evaluation Results==============", rank)
        print_rank_0(f"### Total items\nPositive {scores[3][0]} Hard {scores[3][1]} Easy {scores[3][2]} All {sum(scores[3])}", rank)
        print_rank_0(f"### Accuracy\nPositive-Hard Negative: {scores[0][0] * 100:.2f}", rank)
        print_rank_0(f"Positive-Easy Negative: {scores[0][1] * 100:.2f}", rank)
        print_rank_0(f"Hard Negative-Easy Negative: {scores[0][2] * 100:.2f}", rank)
        print_rank_0(f"Overall: {total_acc * 100:.2f}", rank)
        print_rank_0(f"### Average Scores\nPositive: {scores[2][0]:.2f}", rank)
        print_rank_0(f"Hard Negative: {scores[2][1]:.2f}", rank)
        print_rank_0(f"Easy Negative: {scores[2][2]:.2f}", rank)
    except:
        import traceback
        print_rank(f"Get Evaluation {traceback.format_exc()}", rank)


def main():
    # 1 预处理    
    args = parse_args()
    config = P2QConfig.init_from_yaml(args.config_path)
    batch_size = config.per_device_train_batch_size
    rank = config.local_rank = args.local_rank
    if rank == -1:
        rank = os.environ.get("LOCAL_RANK", "")
        if (isinstance(rank, int) or rank.isnumeric()) and int(rank) >= 0: config.local_rank = rank = int(os.environ['LOCAL_RANK'])
        else: rank = -1
    ds_stage = config.deepspeed_config['zero_optimization']['stage'] if rank >= 0 and config.deepspeed_config['enabled'] else 0
    if rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        if config.deepspeed_config['enabled']: 
            deepspeed.init_distributed()
            ds_config = config.deepspeed_config
            ds_config['find_unused_parameters'] = False
            ds_config['train_micro_batch_size_per_gpu'] = batch_size * 3
            ds_config['train_batch_size'] = batch_size * 3 * dist.get_world_size() * config.gradient_accumulation
    print_rank_0(config, rank)
    set_random_seed(config.seed)
    barrier()

    # 2 tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, fast_tokenizer=True)
    rm_model = AutoModelForSequenceClassification.from_pretrained(config.model_name_or_path, num_labels=1)
    from transformers import DebertaV2ForSequenceClassification
    print_rank_0("Inited RM", rank)
    barrier()

    # 3 加载数据集\
    train_dataset, eval_dataset = load_train_eval_data(config.data_path, config.data_split_test_sets, config.data_min_score_diff)
    train_sampler = RandomSampler(train_dataset) if rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, collate_fn=P2QDataCollator(tokenizer), sampler=train_sampler, batch_size=batch_size)
    print_rank_0("Created prompt dataset", rank)
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(rm_model, config.weight_decay)
    AdamOptimizer = torch.optim.Adam if rank == -1 else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters, lr=config.learning_rate, betas=(0.9, 0.999))
    num_update_steps = math.ceil(len(train_dataloader) / config.gradient_accumulation) * config.train_epochs
    num_warmup_steps = int(config.warmup_ratio * num_update_steps)
    print_rank_0(f"Train length: {len(train_dataloader)} Total steps: {num_update_steps} Warmup steps: {num_warmup_steps}", rank)
    lr_scheduler = get_scheduler(config.warmup_type, optimizer, num_warmup_steps, num_update_steps)
    if config.gradient_checkpointing:
        rm_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if rank >= 0:
        if config.deepspeed_config['enabled']:
            rm_model, optimizer, _, lr_scheduler = deepspeed.initialize(args, rm_model, optimizer, config=ds_config, 
                                                                        lr_scheduler=lr_scheduler, dist_init_required=True)
        else:
            rm_model.to(device)
            rm_model = DDP(rm_model, device_ids=[rank], output_device=[rank])
    # Train!
    print_rank_0(f"***** Evaluating reward, Epoch 0/{config.train_epochs} *****", rank)
    for epoch in range(config.train_epochs):
        print_rank_0(f"Beginning of Epoch {epoch+1}/{config.train_epochs}, Total Micro Batches {len(train_dataloader)}", rank)
        rm_model.train()
        loss_10 = 0
        if isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            rm_logits = rm_model(**batch).logits
            rm_logits = rm_logits.squeeze(1).view(-1, 3)
            ph = torch.clamp(torch.sigmoid(rm_logits[:, 0] - rm_logits[:, 1]), 1e-8, 1-1e-8)
            hn = torch.clamp(torch.sigmoid(rm_logits[:, 1] - rm_logits[:, 2]), 1e-8, 1-1e-8)
            rm_loss = -(torch.log(ph) + torch.log(hn)).mean() / 2
            loss_10 += rm_loss.item()
            rm_model.backward(rm_loss)
            rm_model.step()
            if (step + 1) % 10 == 0:
                print_rank_0(f"Epoch {epoch+1}/{config.train_epochs}|Step {step+1}/{len(train_dataloader)}|loss {loss_10/10:.4f}", rank)
                loss_10 = 0
        print_rank_0(f"***** Evaluating reward, Epoch {epoch+1}/{config.train_epochs} *****", rank)
        validate(rm_model, tokenizer, config, eval_dataset, rank, device)
        rm_model.tput_timer.update_epoch_count()
        if config.model_save_path:
            save_model(rm_model, config, tokenizer, ds_stage, f"epoch{epoch}")


if __name__ == "__main__":
    main()
