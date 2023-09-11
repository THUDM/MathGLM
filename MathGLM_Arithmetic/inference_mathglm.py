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

from SwissArmyTransformer import get_args
from SwissArmyTransformer.model import CachedAutoregressiveModel
from SwissArmyTransformer.generation.sampling_strategies import BaseStrategy
from SwissArmyTransformer.generation.autoregressive_sampling import filling_sequence
from SwissArmyTransformer.generation.utils import timed_name, generate_continually
from icetk import icetk



def main(args):
    # load model from saved checkpoint
    model_path = '/path/to/checkponts/'
    
    model, args = CachedAutoregressiveModel.from_pretrained(args, model_path)
    
    # define function for each query
    end_tokens = [icetk.encode('SEP')[-1]]
    strategy = BaseStrategy(temperature=args.temperature, top_k=args.top_k, end_tokens=end_tokens)
    icetk.text_tokenizer.discourage_tokens(['▁[', '▁('])
    icetk.text_tokenizer.discourage_tokens(['▁[', '▁(', '▁+', '▁=', '▁*', '▁-'])
    icetk.text_tokenizer.discourage_ids(range(125653,130000))
    def process(raw_text): 
        if args.with_id:
            query_id, raw_text = raw_text.split('\t')
        
        """data format conversion"""
        if '*' in raw_text or '-' in raw_text or '+' in raw_text or '/' in raw_text:
            raw_text = raw_text.replace(" ","")
            if '=' not in raw_text:
                raw_text += '='
        seq = icetk.encode(raw_text)
        if seq[0] == 20005:
            seq = seq[1:]
        seq += [-1] * (args.max_sequence_length - len(seq))
        if len(seq) > 1024:
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
        output_tokens = output_list
        # find SEP to obatin output 
        for i in range(len(output_list)):
            output = output_list[i].tolist() 
            try:
                unfinished = output.index(-1)
            except ValueError:
                unfinished = len(output)
            # print("unfinished@@@", unfinished)
            if output[unfinished - 1] in end_tokens:
                unfinished -= 1
            output_list[i] = output[:unfinished]

        # decoding 
        txts = [] 
        for seq in output_tokens:
            decoded_txts = icetk.decode(seq)
            txts.append(decoded_txts)
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
    py_parser.add_argument('--full-query', action='store_true')

    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    
    with torch.no_grad():
        main(args)

