export CUDA_VISIBLE_DEVICES=0
## 51 서버에서 Oracle 세팅 => Instruct BLIP inference로 바꿔끼기!
## 1. oracle 세팅 재구현 되는지 확인. (full로만 확인)
## 2. blip output갈아끼우기
IMAGE_DIR="/nfs_data_storage/mimic-cxr-jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
DATA_DIR="/nfs_edlab/shcho/lec/idl/project/data/"
DESCRIPTION_PATH="/nfs_edlab/shcho/multimodal_rrs/proposed_llama/blip/prompt/disease_descriptor_specific_11.json"

SAVE_BLIP_PATH="/nfs_edlab/shcho/multimodal_rrs/proposed_llama/blip/generation/output_blip_gpt_specific_11_{}.json"
SAVE_LLAMA_PROMPT_PATH="/nfs_edlab/shcho/multimodal_rrs/proposed_llama/blip/llama_prompt/oracle_8_reimpl_{}_prompt.json"
# SAVE_LLAMA_PATH="/nfs_edlab/shcho/multimodal_rrs/proposed_llama/llama/output/llama_blip_gpt_oracle_8_{}_result.json"
SAVE_LLAMA_PATH="/nfs_edlab/shcho/multimodal_rrs/proposed_llama/llama/output/oracle_8_reimpl_{}_result.json"

AGG_TYPE="none" # or avg, sum


cd blip
# python blip.py \
# --data_type full \
# --use_oracle true \
# --image_dir $IMAGE_DIR --data_dir $DATA_DIR \
# --instruct_blip_path /nfs_data_storage/instruct_blip \
# --description_path $DESCRIPTION_PATH \
# --save_file_path $SAVE_BLIP_PATH \
# --save_prompt_path $SAVE_LLAMA_PROMPT_PATH \
# --agg_type $AGG_TYPE

cd ../llama
python llama_amp_with_blip.py \
--data_type all \
--image_dir $IMAGE_DIR --data_dir $DATA_DIR \
--llama_path /nfs_data_storage/llama2_hf/Llama-2-7b-hf/ \
--prompt_path $SAVE_LLAMA_PROMPT_PATH \
--save_file_path $SAVE_LLAMA_PATH

