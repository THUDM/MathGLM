# -*- encoding: utf-8 -*-
'''
@File    :   pretrain_gpt2.py
@Time    :   2021/10/06 00:58:32
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch 
import argparse
import numpy as np

from SwissArmyTransformer import mpu, get_args, get_tokenizer
from SwissArmyTransformer.training.deepspeed_training import training_main
from SwissArmyTransformer import get_args, AutoModel
from SwissArmyTransformer.model.mixins import CachedAutoregressiveMixin
from SwissArmyTransformer.model import GLMModel
from load_dataset import MathDataset

def get_batch(data_iterator, args, timers):
    # Items and their type.
    keys = ['sentence', 'label']
    datatype = torch.int64

    # Broadcast data.
    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    timers('data loader').stop()
    data_b = mpu.broadcast_data(keys, data, datatype)
    # Unpack.
    tokens = data_b['sentence'].long()
    labels = data_b['label'].long()
    batch_size, seq_length = tokens.size()

    position_ids = torch.zeros(2, seq_length, device=tokens.device, dtype=torch.long)
    torch.arange(0, seq_length, out=position_ids[0, :seq_length])
    position_ids = position_ids.unsqueeze(0)

    attention_mask = torch.ones((batch_size, seq_length, seq_length), device=tokens.device)
    attention_mask.tril_()
    attention_mask.unsqueeze_(1)

    # Convert
    if args.fp16:
        attention_mask = attention_mask.half()
    return tokens, labels, attention_mask, position_ids, (tokens!=-1)


def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    tokens, labels, attention_mask, position_ids, loss_mask = get_batch(
        data_iterator, args, timers)
    timers('batch generator').stop()

    logits, *mems = model(tokens, position_ids, attention_mask)
    
    losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(), labels)
    # scaling loss mask
    loss_mask = loss_mask.view(-1)  

    losses = losses.view(-1) * loss_mask
    loss = torch.sum(losses) / loss_mask.sum()
    
    return loss, {}


def create_dataset_function(path, args):
    tokenizer = get_tokenizer(args)
    def process_fn(row):
        ids = []
        row_list = row.split('s')
        for idx, value in enumerate(row_list):
            if idx != len(row_list) - 1:
                value = value
            seq = tokenizer._encode(value)
            if len(seq) != 0 and seq[0] == 20005:
                seq = seq[1:]
            seq = [tokenizer.get_command('ENC').Id] + seq + [tokenizer.get_command('eos').Id]
            ids.extend(seq)
        sentence = ids 
        if len(sentence) < args.sample_length:
            sentence.extend([-1] * (args.sample_length - len(sentence)))
        text_input = sentence[:-1] 
        label = sentence[1:]
        return {'sentence': np.array(text_input, dtype=np.int64), 'label': np.array(label, dtype=np.int64)}
    return MathDataset(path, process_fn, args.sample_length)


if __name__ == '__main__':    
    # The model consists of 28 layers with a model dimension of 4096, and a feedforward dimension of 16384. The model dimension is split into 16 heads, each with a dimension of 256. 
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--new_hyperparam', type=str, default=None)
    py_parser.add_argument('--sample_length', type=int, default=512)
    py_parser.add_argument('--prefix_len', type=int, default=0)
    GLMModel.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args() 
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    model, args = AutoModel.from_pretrained(args, "glm-10b-zh")
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    training_main(args, model_cls=model, forward_step_function=forward_step, create_dataset_function=create_dataset_function)



