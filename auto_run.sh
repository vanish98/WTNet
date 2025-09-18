#!/bin/bash

cd ./src || exit 1
work_dir="transformer"
dataset="YAGO"
export LOG_DIR="../model/${work_dir}/${dataset}/logs"
mkdir -p $LOG_DIR
#multi GPU ctrl+/
#base_command="python -m torch.distributed.launch  --nproc_per_node=4 main.py --gpus 0 1 2 3\
#    -d WIKI --batch_size 6 --n_epoch 15 --lr 0.00005 \
#    --hidden_dims 64 64 64 64 --time_encoding_independent \
#     --num_heads 8 --num_transformer_hiddens 128  --num_layers 4 "
#single GPU
base_command="python  main.py --gpus 0 \
    -d YAGO --batch_size 1 --n_epoch 30 --lr 0.00005 \
    --hidden_dims 64 64 64 64 --time_encoding_independent \
    --windows_size 5 --num_heads 8 --num_transformer_hiddens 128  --num_layers 4 --work_dir ${work_dir}"
parameter_id=11
history_len=10

#12,15,18,19,20
#for history_len in {12,15,18,19,20}; do
#  for hidden_dims in "${hidden_dims_list[@]}"; doK
        log_file="${LOG_DIR}/hlen${history_len}_layers4_layers${parameter_id}.log"
        cmd="$base_command --history_len $history_len --parameter_id $parameter_id "
        echo "Executing: $cmd" | tee -a $log_file
        eval $cmd || { echo "Failed: $cmd"; exit 1; }
#    done
#donei0,