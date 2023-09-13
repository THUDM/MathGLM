## GPT Can Solve Mathematical Problems Without a Calculator <br><sub>Official Pytorch Implementation</sub>

![](resources/perf.jpg)
![](resources/perf_mwp.jpg)
Previous studies have typically assumed that large language models are unable to accurately perform arithmetic operations, particularly multiplication of >8 digits, and operations involving decimals and fractions, without the use of calculator tools. This paper aims to challenge this misconception. With sufficient training data, a 2 billion-parameter language model can accurately perform multi-digit arithmetic operations with almost 100% accuracy without data leakage, significantly surpassing GPT-4 (whose multi-digit multiplication accuracy is only 4.3%). We also demonstrate that our MathGLM, finetuned from GLM-10B on a dataset with additional multi-step arithmetic operations and math problems described in text, achieves similar performance to GPT-4 on a 5,000-samples Chinese math problem
test set.



If you want to find the detailed introduction, Read our paper: [GPT Can Solve Mathematical Problems Without a Calculator](https://arxiv.org/pdf/2309.03241v2.pdf).

# Demo

## Arithmetic Tasks



## Math Word Problem



# Model Download 


## Arithmetic Tasks



## Math Word Problem



# Setup

### Environment
Our MathGLM relies on sat(SwissArmyTransformer), please ``` pip install SwissArmyTransformer ```.

Download the repo and setup the environment with:

```bash
git clone https://github.com/THUDM/MathGLM.git
cd MathGLM
conda env create -f env.yml
conda activate mathglm
```
#### Note
For arithmetic tasks and MathGLM-10B: deepspeed==0.6.0; For math word problems on MathGLM-6B: deepspeed==0.9.5



### Dataset

For arithmetic tasks, please download pre-training dataset from [MathGLM-dataset](https://cloud.tsinghua.edu.cn/d/8d9ee3e52bb54afd9c16/). For amth word problem, the reconstructed Ape210K dataset is provided in ```MathGLM_MWP/dataset/data.jsonl```


## Inference 

### Performance Reproduction

MathGLM achieves competitive results in comparison with the most powerful large language model GPT-4 and ChatGPT.

| Model   | ACC | RE | 
| --------- | ---------- | ---------------- | 
| GPT-4 | 18.84%    | -             |
| ChatGPT  | 10.00%    | -            | 
| MathGLM-10M  | 61.21%    | 97.83%            | 
| MathGLM-100M  | 70.28%    | 99.28%            | 
| MathGLM-500M  | 89.57%    | 99.41%            | 
| MathGLM-2B  | 93.03%    | 99.71%            | 

MathGLM-10B achieves similar performance to GPT-4 on a 5,000-samples Chinese math problem test set.
| Model   | Arithmetic_ACC | Answer_ACC | 
| --------- | ---------- | ---------------- | 
| GPT-4 | -  | 59.57%            |
| ChatGPT  | -   | 39.78%        | 
| MathGLM-Large  | 62.00%   | 50.80%            | 
| MathGLM-GLM-6B  | 64.60%   | 48.06%            | 
| MathGLM-10B  | 69.08%    | 58.68%            | 
| MathGLM-GLM2-6B  | 52.24%   | 45.48%           | 
| MathGLM-ChatGLM-6B  | 58.52%    | 42.28%           | 
| MathGLM-ChatGLM2-6B  | 50.38%    | 43.14%           | 

## Pre-training

For arithmetic tashs, run command:

```bash
cd MathGLM_Arithmetic
./pretrain.sh
```

For math word problem, run command:

```bash
cd MathGLM_MWP
./continue.sh
```


## Citation
