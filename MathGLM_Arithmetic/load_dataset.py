# -*- encoding: utf-8 -*-
'''
@File    :   datasets.py
@Time    :   2021/01/11 21:01:51
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
from functools import partial
import os
import sys
import math
import random

import numpy as np
import pickle
import time
import json

import torch
from torch.utils.data import Dataset

# load dataset for MathGLM
class MathDataset(Dataset):
    def __init__(self, path, process_fn, length_per_sample, dtype='int32', preload=False, **kwargs): # TODO ARGS
        t_process = time.time()
        self.dtype = np.dtype(dtype)
        self.process_fn = process_fn
        self.length_per_sample = length_per_sample
        with open(path, 'r') as fin:
            inputs = fin.read().splitlines()
        # len constraint
        inputs_all = []
        for line in inputs:
            if len(line) < self.length_per_sample:
                inputs_all.append(line)
        print("inputs_all", len(inputs_all))
        
        inputs = 's'.join(inputs_all)
        self.math = []
        start = 0
        end = self.length_per_sample
        while end < len(inputs):
            seq = inputs[start:end]
            if len(seq) != self.length_per_sample:
                break
            if seq[-1] != 's':
                index = seq.rindex('s')
                start = start + index + 1 
            else:
                start = end
            end = start + self.length_per_sample
            if end > len(inputs):
                end = len(inputs)
            self.math.append(seq)
        print("t_process......", time.time() - t_process)
        print("math_length", len(self.math))
        
    def __len__(self):
        return len(self.math)
    
    def __getitem__(self, index):
        return self.process_fn(self.math[index])
    



