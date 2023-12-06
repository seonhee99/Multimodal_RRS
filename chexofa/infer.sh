Generating summary with pre-trained models
export GPU=5

export DATA_NAME="mimic-cxr-images-512"
export MODEL_NAME="base-coco"

export CKPT_NM="corrupted_02_dataset"
export ROOT_DIR="/home/jiho/CMU_IntroDL/project/dataset"
export DATA_PATH=${ROOT_DIR}/${DATA_NAME}

export TOKENIZER_PATH="/home/jiho/CMU_IntroDL/chexofa-master/results/jiho2" 
export MODEL_PATH="/home/jiho/CMU_IntroDL/chexofa-master/results/jiho2"   
export RUN_NAME=${CKPT_NM}

export MIN_LEN=10
export MAX_LEN=400 #100
export TEMPER=1.0
export LEN_PEN=0.5

export OUT_DIR="/home/jiho/CMU_IntroDL/chexofa-master/rsum_results"
export OUT_PATH=$OUT_DIR/$RUN_NAME/len-$MIN_LEN-$MAX_LEN-temp$TEMPER-len$LEN_PEN

CUDA_VISIBLE_DEVICES=${GPU} python ./run_summarization.py \
    --image_dir ${DATA_PATH}/files/ \
    --ann_path "/home/jiho/CMU_IntroDL/project/Rsum_annotation_corrupted_0.2.json" \
    --tokenizer_path ${TOKENIZER_PATH} \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ${OUT_PATH} \
    --do_eval --do_eval_w_valid \
    --num_workers 30 \
    --dataset_name 'mimic_cxr_summarization' \
    --max_seq_length 400 \
    --per_device_eval_batch_size 12 \
    --min_len $MIN_LEN \
    --max_len $MAX_LEN \
    --temperature ${TEMPER} \
    --length_penalty ${LEN_PEN} \
	--use_wandb \
    --wandb_project_name eval_CheXOFA_summary \
    --wandb_run_name ${RUN_NAME}