# CheXOFA


## Radiology Report Generation
### MIMIC-CXR

export GPU=7

export DATA_NAME="mimic-cxr-images-512" 
export OUT_NAME="coco"
export ROOT_DIR="/home/jiho/CMU_IntroDL/project/dataset"
export DATA_PATH=${ROOT_DIR}/${DATA_NAME}
export MODEL_PATH="/home/jiho/CMU_IntroDL/chexofa-master/transformers/src/transformers/models/ofa/ofa-base-coco" 

export LR=1e-5
export DECAY=0.01
export EPOCH=10
export TRAIN_BATCH_SIZE=24

# /home/jiho/CMU_IntroDL/project/idl_rgen_annotation.json
# ${ROOT_DIR}/report_generation_annotations.json \

export OUT_PATH="results/jiho" #${OUT_NAME}-lr${LR}

CUDA_VISIBLE_DEVICES=$GPU python ./run_caption.py \
    --image_dir ${DATA_PATH}/files/ \
    --ann_path /home/jiho/CMU_IntroDL/project/idl_rgen_annotation.json\
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ${OUT_PATH} \
    --do_train \
    --do_valid \
    --do_eval \
    --num_workers 30 \
    --dataset_name mimic_cxr \
    --max_seq_length 400 \
    --per_device_train_batch_size ${TRAIN_BATCH_SIZE} \
    --num_train_epochs ${EPOCH} \
    --learning_rate ${LR} \
    --weight_decay ${DECAY} \
    --per_device_eval_batch_size 12 \
    --min_len 30 \
    --max_len 400 \
    --use_wandb \
    --valid_steps 10 \
    --logging_steps 5
