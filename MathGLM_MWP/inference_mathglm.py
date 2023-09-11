# -*- encoding: utf-8 -*-
'''
@File    :   inference_cogview.py
@Time    :   2021/10/09 19:41:58
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
import stat

from SwissArmyTransformer import mpu, get_args, get_tokenizer
from SwissArmyTransformer.model import CachedAutoregressiveModel
from SwissArmyTransformer.generation.sampling_strategies import BaseStrategy
from SwissArmyTransformer.generation.autoregressive_sampling import filling_sequence
from SwissArmyTransformer.generation.utils import timed_name, generate_continually
from SwissArmyTransformer.training import set_random_seed

import json

def main(args):

    '''
    2022/06/17
    Modify load_checkpoint to from_pretraind
    '''
    # initialize_distributed(args)
    # load model from saved checkpoint 
    
    model_path = '/path/to/checkpoints/'
    
    model, args = CachedAutoregressiveModel.from_pretrained(args, model_path)

    if args.fp16:
        model = model.half()
    model = model.to(args.device)
    set_random_seed(args.seed)
    model.eval()
    
    tokenizer = get_tokenizer(args)
 
    # define function for each query
    end_tokens = [tokenizer.get_command('eos').Id] 
    strategy = BaseStrategy(temperature=args.temperature, top_k=args.top_k, end_tokens=end_tokens)
    
    def process(raw_text): 
        if args.with_id:
            query_id, raw_text = raw_text.split('\t')
        raw_text = json.loads(raw_text)
        question=raw_text["question"] + "答："
        raw_text = question
        seq = tokenizer._encode(raw_text)
        if len(seq) != 0 and seq[0] == 20005:
            seq = seq[1:]
        seq = [tokenizer.get_command('ENC').Id] + seq
        seq += [-1] * (args.max_sequence_length - len(seq)) 
        if len(seq) > args.max_sequence_length:
            raise ValueError('text too long.')
        # generation
        seq = torch.cuda.LongTensor(seq, device=args.device)
        mbz = args.max_inference_batch_size
        assert args.batch_size < mbz or args.batch_size % mbz == 0
        output_list = []
        for tim in range(max(args.batch_size // mbz, 1)):
            output = filling_sequence(model, seq.clone(),
                    batch_size=min(args.batch_size, mbz),
                    strategy=strategy,
                    log_attention_weights=None
                    )[0]
            if isinstance(output, torch.Tensor): # different strategies
                output = list(output)

            output_list.extend(output)
        # find SEP to obatin output 
        for i in range(len(output_list)):
            output = output_list[i].tolist() 
            try:
                unfinished = output.index(-1)
            except ValueError:
                unfinished = len(output)
            if output[unfinished - 1] in end_tokens:
                unfinished -= 1
            output_list[i] = output[1:unfinished]
            bog = output.index(tokenizer.get_command('eos').Id)
            output_list[i] = output[1:bog] + output[bog+1:unfinished] 
                
        # decoding 
        txts = [] 
        for seq in output_list:
            decode_tokens = tokenizer.DecodeIds(seq)
            txts.append(decode_tokens)
        
        # save
        if args.with_id:
            full_path = os.path.join(args.output_path, query_id + '.txt')
        else:
            prefix = raw_text.replace('/', '')[:20]
            full_path = timed_name(prefix, '.txt', args.output_path)
            print(txts[0]) # print the first.
        test_eval_path = os.path.join(args.output_path, 'test_eval.txt')
        with open(test_eval_path, 'a', encoding='utf-8') as fout:
            fout.write(txts[0] + '\n')
        os.chmod(test_eval_path, stat.S_IRWXO + stat.S_IRWXG + stat.S_IRWXU)

    os.makedirs(args.output_path, exist_ok=True)
    generate_continually(process, args.input_source) 


if __name__ == "__main__":
    py_parser = argparse.ArgumentParser(add_help=False)
    
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    args.do_train = False
    
    with torch.no_grad():
        main(args)

