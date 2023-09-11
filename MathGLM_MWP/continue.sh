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

full_data="/path/to/dataset/data.jsonl" 


gpt_options=" \
       --experiment-name finetune-gpt-j-mathglm \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --batch-size 16 \
       --train-iters 30000 \
       --resume-dataloader \
       --train-data ${full_data} \
       --split 1  \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .01 \
       --checkpoint-activations \
       --fp16 \
       --zero-stage 2 \
       --save-interval 1000 \
       --eval-interval 1000 \
       --save /path/to/checkpoints \
"

run_cmd="${OPTIONS_NCCL} deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 5600 --hostfile ${HOST_FILE_PATH} continue_train_mathglm.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x


