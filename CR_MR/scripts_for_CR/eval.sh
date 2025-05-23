

CUDA_VISIBLE_DEVICES=$2 python commonsense_evaluate.py \
    --model Qwen2.5-7B-Instruct \
    --adapter LoRA \
    --dataset boolq \
    --base_model 'Qwen/Qwen2.5-7B-Instruct' \
    --batch_size 1 \
    --lora_weights $1|tee -a $1/boolq.txt

CUDA_VISIBLE_DEVICES=$2 python commonsense_evaluate.py \
    --model Qwen2.5-7B-Instruct \
    --adapter LoRA \
    --dataset piqa \
    --base_model 'Qwen/Qwen2.5-7B-Instruct' \
    --batch_size 1 \
    --lora_weights $1|tee -a $1/piqa.txt

CUDA_VISIBLE_DEVICES=$2 python commonsense_evaluate.py \
    --model Qwen2.5-7B-Instruct \
    --adapter LoRA \
    --dataset social_i_qa \
    --base_model 'Qwen/Qwen2.5-7B-Instruct' \
    --batch_size 1 \
    --lora_weights $1|tee -a $1/social_i_qa.txt

CUDA_VISIBLE_DEVICES=$2 python commonsense_evaluate.py \
    --model Qwen2.5-7B-Instruct \
    --adapter LoRA \
    --dataset hellaswag \
    --base_model 'Qwen/Qwen2.5-7B-Instruct' \
    --batch_size 1 \
    --lora_weights $1|tee -a $1/hellaswag.txt

CUDA_VISIBLE_DEVICES=$2 python commonsense_evaluate.py \
    --model Qwen2.5-7B-Instruct \
    --adapter LoRA \
    --dataset winogrande \
    --base_model 'Qwen/Qwen2.5-7B-Instruct' \
    --batch_size 1 \
    --lora_weights $1|tee -a $1/winogrande.txt

CUDA_VISIBLE_DEVICES=$2 python commonsense_evaluate.py \
    --model Qwen2.5-7B-Instruct \
    --adapter LoRA \
    --dataset ARC-Challenge \
    --base_model 'Qwen/Qwen2.5-7B-Instruct' \
    --batch_size 1 \
    --lora_weights $1|tee -a $1/ARC-Challenge.txt

CUDA_VISIBLE_DEVICES=$2 python commonsense_evaluate.py \
    --model Qwen2.5-7B-Instruct \
    --adapter LoRA \
    --dataset ARC-Easy \
    --base_model 'Qwen/Qwen2.5-7B-Instruct' \
    --batch_size 1 \
    --lora_weights $1|tee -a $1/ARC-Easy.txt

CUDA_VISIBLE_DEVICES=$2 python commonsense_evaluate.py \
    --model Qwen2.5-7B-Instruct \
    --adapter LoRA \
    --dataset openbookqa \
    --base_model 'Qwen/Qwen2.5-7B-Instruct' \
    --batch_size 1 \
    --lora_weights $1|tee -a $1/openbookqa.txt
