
## Radiology Report Summarization

# Debugging with toy dataset
```
export GPU=6

export DATA_NAME="mimic-cxr-images-512" 
export OUT_NAME="R_sum_jiho"

export ROOT_DIR="/home/jiho/CMU_IntroDL/project/dataset" 
export DATA_PATH=${ROOT_DIR}/${DATA_NAME}
export MODEL_PATH="/home/jiho/CMU_IntroDL/chexofa-master/transformers/src/transformers/models/ofa/ofa-base-coco2" 

export LR=5e-5
export DECAY=0.01
export EPOCH=5
export TRAIN_BATCH_SIZE=16

export OUT_PATH="results/jiho2" 

CUDA_VISIBLE_DEVICES=$GPU python ./run_summarization.py \
    --image_dir ${DATA_PATH}/files/ \
    --ann_path "/home/jiho/CMU_IntroDL/project/Rsum_annotation_jioh.json"\
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ${OUT_PATH} \
    --do_train \
    --do_valid \
    --do_eval \
    --num_workers 30 \
    --dataset_name 'mimic_cxr_summarization' \
    --max_seq_length 400 \
    --per_device_train_batch_size ${TRAIN_BATCH_SIZE} \
    --num_train_epochs ${EPOCH} \
    --learning_rate ${LR} \
    --weight_decay ${DECAY} \
    --per_device_eval_batch_size 12 \
    --min_len 5 \
    --max_len 100 \
    --use_wandb \
    --valid_steps 100 \
    --logging_steps 20
```
