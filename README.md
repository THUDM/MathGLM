## GPT Can Solve Mathematical Problems Without a Calculator <br><sub>Official Pytorch Implementation</sub>

![](resources/perf.jpg)
![](resources/perf_mwp.jpg)
Previous studies have typically assumed that large language models are unable to accurately perform arithmetic operations, particularly multiplication of >8 digits, and operations involving decimals and fractions, without the use of calculator tools. This paper aims to challenge this misconception. With sufficient training data, a 2 billion-parameter language model can accurately perform multi-digit arithmetic operations with almost 100% accuracy without data leakage, significantly surpassing GPT-4 (whose multi-digit multiplication accuracy is only 4.3%). We also demonstrate that our MathGLM, finetuned from GLM-10B on a dataset with additional multi-step arithmetic operations and math problems described in text, achieves similar performance to GPT-4 on a 5,000-samples Chinese math problem
test set.



If you want to find the detailed introduction, Read our paper: [GPT Can Solve Mathematical Problems Without a Calculator](https://arxiv.org/pdf/2309.03241.pdf).


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

### Dataset
