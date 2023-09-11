#!/bin/bash
# SAT_HOME=/raid/dm/sat_models

NLAYERS=48
NHIDDEN=2560
NATT=40
MAXSEQLEN=1089
MPSIZE=1

#SAMPLING ARGS
# TEMP=1.03
TEMP=0.1
TOPK=200

export CUDA_VISIBLE_DEVICES=7
# SAT_HOME=$SAT_HOME \
python inference_mathglm.py \
       --mode inference \
       --distributed-backend nccl \
       --max-sequence-length 512 \
       --fp16 \
       --model-parallel-size $MPSIZE \
       --num-layers $NLAYERS \
       --hidden-size $NHIDDEN \
       --num-attention-heads $NATT \
       --temperature $TEMP \
       --top_k $TOPK \
       --input-source ./input_test.txt \
       --output-path samples_result \
       --batch-size 1 \
       --max-inference-batch-size 8 \
       $@


