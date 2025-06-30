

CUDA_VISIBLE_DEVICES=$2 python math_evaluate.py \
    --model Qwen2.5-7B-Instruct \
    --adapter LoRA \
    --dataset gsm8k \
    --base_model 'Qwen/Qwen2.5-7B-Instruct' \
    --batch_size 20 \
    --lora_weights $1|tee -a $1/gsm8k.txt

# CUDA_VISIBLE_DEVICES=$2 python math_evaluate.py \
#     --model Qwen2.5-7B-Instruct \
#     --adapter LoRA \
#     --dataset AQuA \
#     --base_model 'Qwen/Qwen2.5-7B-Instruct' \
#     --batch_size 20 \
#     --lora_weights $1|tee -a $1/AQuA.txt

# CUDA_VISIBLE_DEVICES=$2 python math_evaluate.py \
#     --model Qwen2.5-7B-Instruct \
#     --adapter LoRA \
#     --dataset MultiArith \
#     --base_model 'Qwen/Qwen2.5-7B-Instruct' \
#     --batch_size 20 \
#     --lora_weights $1|tee -a $1/MultiArith.txt

# CUDA_VISIBLE_DEVICES=$2 python math_evaluate.py \
#     --model Qwen2.5-7B-Instruct \
#     --adapter LoRA \
#     --dataset SVAMP \
#     --base_model 'Qwen/Qwen2.5-7B-Instruct' \
#     --batch_size 20 \
#     --lora_weights $1|tee -a $1/SVAMP.txt

# CUDA_VISIBLE_DEVICES=$2 python math_evaluate.py \
#     --model Qwen2.5-7B-Instruct \
#     --adapter LoRA \
#     --dataset SingleEq \
#     --base_model 'Qwen/Qwen2.5-7B-Instruct' \
#     --batch_size 20 \
#     --lora_weights $1|tee -a $1/SingleEq.txt

# CUDA_VISIBLE_DEVICES=$2 python math_evaluate.py \
#     --model Qwen2.5-7B-Instruct \
#     --adapter LoRA \
#     --dataset AddSub \
#     --base_model 'Qwen/Qwen2.5-7B-Instruct' \
#     --batch_size 20 \
#     --lora_weights $1|tee -a $1/AddSub.txt
