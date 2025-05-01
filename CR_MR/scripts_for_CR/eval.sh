

    CUDA_VISIBLE_DEVICES=$2 python commonsense_evaluate.py \
    --model Qwen2.5-7B \
    --adapter LoRA \
    --dataset social_i_qa \
    --base_model 'Qwen/Qwen2.5-7B' \
    --batch_size 1 \
    --lora_weights $1|tee -a $1/social_i_qa.txt