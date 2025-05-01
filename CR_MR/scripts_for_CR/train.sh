CUDA_VISIBLE_DEVICES=$4 python finetune.py \
    --base_model 'Qwen/Qwen2.5-7B' \
    --data_path 'commonsense_170k.json' \
    --output_dir $3 \
    --batch_size 16  --micro_batch_size 16 --num_epochs 3 \
    --learning_rate 1e-4 --cutoff_len 256 --val_set_size 120 \
    --eval_step 80 --save_step 80  --adapter_name lora \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --lora_r $1 --lora_alpha $2 --use_gradient_checkpointing