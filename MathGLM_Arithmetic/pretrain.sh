#! /bin/bash

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=1
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

OPTIONS_NCCL="NCCL_DEBUG=warning NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
HOST_FILE_PATH="hostfile"
HOST_FILE_PATH="hostfile_single"

full_data="/path/to/datasets/dataset.txt"


#  500M 1024*1024*40
gpt_options=" \
       --experiment-name pretrain-gpt2-mathglm \
       --model-parallel-size ${MP_SIZE} \
       --mode pretrain \
       --num-layers 40 \
       --hidden-size 1024 \
       --num-attention-heads 32 \
       --batch-size 32 \
       --train-iters  1000000 \
       --vocab-size 61667 \
       --resume-dataloader \
       --checkpoint-activations \
       --train-data ${full_data} \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .01 \
       --max-sequence-length 1024 \
       --fp16 \
       --zero-stage 1 \
       --save-interval 5000 \
       --eval-interval 1000 \
       --save /path/to/checkpoints \
"



#  2B 2048*2048*40
# gpt_options=" \
#        --experiment-name pretrain-gpt2-mathglm \
#        --model-parallel-size ${MP_SIZE} \
#        --mode pretrain \
#        --num-layers 40 \
#        --hidden-size 2048 \
#        --num-attention-heads 32 \
#        --batch-size 32 \
#        --train-iters  1000000 \
#        --vocab-size 61667 \
#        --resume-dataloader \
#        --checkpoint-activations \
#        --train-data ${full_data} \
#        --split 949,50,1 \
#        --distributed-backend nccl \
#        --lr-decay-style cosine \
#        --warmup .01 \
#        --max-sequence-length 896 \
#        --fp16 \
#        --zero-stage 1 \
#        --save-interval 5000 \
#        --eval-interval 1000 \
#        --save /path/to/checkpoints \
# "


# 10M: 15*256*256
# gpt_options=" \
#        --experiment-name pretrain-gpt2-mathglm \
#        --model-parallel-size ${MP_SIZE} \
#        --mode pretrain \
#        --num-layers 15 \
#        --hidden-size 256 \
#        --num-attention-heads 32 \
#        --batch-size 64 \
#        --train-iters  1000000 \
#        --vocab-size 61667 \
#        --resume-dataloader \
#        --train-data ${full_data} \
#        --split 949,50,1 \
#        --distributed-backend nccl \
#        --lr-decay-style cosine \
#        --warmup .01 \
#        --checkpoint-activations \
#        --max-sequence-length 512 \
#        --fp16 \
#        --zero-stage 1 \
#        --save-interval 5000 \
#        --eval-interval 1000 \
#        --save /path/to/checkpoints \
# "



# 100M 35*512*512
# gpt_options=" \
#        --experiment-name pretrain-gpt2-mathglm \
#        --model-parallel-size ${MP_SIZE} \
#        --mode pretrain \
#        --num-layers 35 \
#        --hidden-size 512 \
#        --num-attention-heads 32 \
#        --batch-size 64 \
#        --train-iters  1000000 \
#        --vocab-size 61667 \
#        --resume-dataloader \
#        --train-data ${full_data} \
#        --split 949,50,1 \
#        --distributed-backend nccl \
#        --lr-decay-style cosine \
#        --warmup .01 \
#        --checkpoint-activations \
#        --max-sequence-length 512 \
#        --fp16 \
#        --zero-stage 1 \
#        --save-interval 5000 \
#        --eval-interval 1000 \
#        --save /path/to/checkpoints \
# "


run_cmd="${OPTIONS_NCCL} deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 62000 --hostfile ${HOST_FILE_PATH} pretrain_mathglm.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
