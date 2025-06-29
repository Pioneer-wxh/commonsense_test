# 评估模型在 GSM8K、AQuA、MultiArith、SVAMP、SingleEq 和 AddSub 数据集上的表现
# 参数说明：
# $1: LoRA 的 r 参数
# $2: LoRA 的 alpha 参数
# $3: 输出目录
# $4: 使用的 GPU 设备号

CUDA_VISIBLE_DEVICES=$4 python math_evaluate.py \
    --model Qwen2.5-7B-Instruct \
    --adapter LoRA \
    --dataset gsm8k \
    --base_model 'Qwen/Qwen2.5-7B-Instruct' \
    --lora_weights $3|tee -a $3/gsm8k.txt

CUDA_VISIBLE_DEVICES=$4 python math_evaluate.py \
    --model Qwen2.5-7B-Instruct \
    --adapter LoRA \
    --dataset AQuA \
    --base_model 'Qwen/Qwen2.5-7B-Instruct' \
    --lora_weights $3|tee -a $3/AQuA.txt

CUDA_VISIBLE_DEVICES=$4 python math_evaluate.py \
    --model Qwen2.5-7B-Instruct \
    --adapter LoRA \
    --dataset MultiArith \
    --base_model 'Qwen/Qwen2.5-7B-Instruct' \
    --lora_weights $3|tee -a $3/MultiArith.txt

CUDA_VISIBLE_DEVICES=$4 python math_evaluate.py \
    --model Qwen2.5-7B-Instruct \
    --adapter LoRA \
    --dataset SVAMP \
    --base_model 'Qwen/Qwen2.5-7B-Instruct' \
    --lora_weights $3|tee -a $3/SVAMP.txt

CUDA_VISIBLE_DEVICES=$4 python math_evaluate.py \
    --model Qwen2.5-7B-Instruct \
    --adapter LoRA \
    --dataset SingleEq \
    --base_model 'Qwen/Qwen2.5-7B-Instruct' \
    --lora_weights $3|tee -a $3/SingleEq.txt

CUDA_VISIBLE_DEVICES=$4 python math_evaluate.py \
    --model Qwen2.5-7B-Instruct \
    --adapter LoRA \
    --dataset AddSub \
    --base_model 'Qwen/Qwen2.5-7B-Instruct' \
    --lora_weights $3|tee -a $3/AddSub.txt
