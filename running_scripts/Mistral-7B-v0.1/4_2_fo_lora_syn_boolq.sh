#!/bin/bash -l

# polaris 60h
cd /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models

OUT_0=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/BOOLQ/fo_lora/rejection_sampling/1e-4_lora_rk8.out
ERR_0=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/BOOLQ/fo_lora/rejection_sampling/1e-4_lora_rk8.err
MODEL=mistralai/Mistral-7B-v0.1 MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/BOOLQ LR=1e-4 BS=2 RANK=8 STEPS=160000 SEED=0 bash fo_lora_finetune_synthetic.sh 1>>$OUT_0 2>>$ERR_0 &

wait

OUT_1=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/BOOLQ/fo_lora/rejection_sampling/2e-4_lora_rk8.out
ERR_1=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/BOOLQ/fo_lora/rejection_sampling/2e-4_lora_rk8.err
MODEL=mistralai/Mistral-7B-v0.1 MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/BOOLQ LR=2e-4 BS=2 RANK=8 STEPS=160000 SEED=0 bash fo_lora_finetune_synthetic.sh 1>>$OUT_1 2>>$ERR_1 &

wait

# modify bs
OUT_2=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/BOOLQ/fo_lora/rejection_sampling/1e-4_lora_rk16.out
ERR_2=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/BOOLQ/fo_lora/rejection_sampling/1e-4_lora_rk16.err
MODEL=mistralai/Mistral-7B-v0.1 MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/BOOLQ LR=1e-4 BS=4 RANK=16 STEPS=80000 SEED=0 bash fo_lora_finetune_synthetic.sh 1>>$OUT_2 2>>$ERR_2 &

wait

OUT_3=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/BOOLQ/fo_lora/rejection_sampling/2e-4_lora_rk16.out
ERR_3=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/BOOLQ/fo_lora/rejection_sampling/2e-4_lora_rk16.err
MODEL=mistralai/Mistral-7B-v0.1 MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/BOOLQ LR=2e-4 BS=2 RANK=16 STEPS=160000 SEED=0 bash fo_lora_finetune_synthetic.sh 1>>$OUT_3 2>>$ERR_3 &

wait

# # siai
# cd /home/jlong1/Downloads/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models

# OUT_0=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/rejection_sampling/3_first_order_lora/result/BOOLQ/llama_3.2_1b/1e-4_lora_rk8.out
# ERR_0=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/rejection_sampling/3_first_order_lora/result/BOOLQ/llama_3.2_1b/1e-4_lora_rk8.err
# CUDA_VISIBLE_DEVICES=0 MODEL=mistralai/Mistral-7B-v0.1 MODE=ft TASK=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/BOOLQ LR=1e-4 BS=2 RANK=8 STEPS=160000 SEED=0 bash fo_lora_finetune_synthetic.sh 1>>$OUT_0 2>>$ERR_0 &

# OUT_2=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/rejection_sampling/3_first_order_lora/result/BOOLQ/llama_3.2_1b/1e-4_lora_rk16.out
# ERR_2=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/rejection_sampling/3_first_order_lora/result/BOOLQ/llama_3.2_1b/1e-4_lora_rk16.err
# CUDA_VISIBLE_DEVICES=1 MODEL=mistralai/Mistral-7B-v0.1 MODE=ft TASK=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/BOOLQ LR=1e-4 BS=2 RANK=16 STEPS=160000 SEED=0 bash fo_lora_finetune_synthetic.sh 1>>$OUT_2 2>>$ERR_2 &

# wait

# OUT_1=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/rejection_sampling/3_first_order_lora/result/BOOLQ/llama_3.2_1b/2e-4_lora_rk8.out
# ERR_1=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/rejection_sampling/3_first_order_lora/result/BOOLQ/llama_3.2_1b/2e-4_lora_rk8.err
# CUDA_VISIBLE_DEVICES=0 MODEL=mistralai/Mistral-7B-v0.1 MODE=ft TASK=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/BOOLQ LR=2e-4 BS=2 RANK=8 STEPS=160000 SEED=0 bash fo_lora_finetune_synthetic.sh 1>>$OUT_1 2>>$ERR_1 &

# OUT_3=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/rejection_sampling/3_first_order_lora/result/BOOLQ/llama_3.2_1b/2e-4_lora_rk16.out
# ERR_3=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/rejection_sampling/3_first_order_lora/result/BOOLQ/llama_3.2_1b/2e-4_lora_rk16.err
# CUDA_VISIBLE_DEVICES=1 MODEL=mistralai/Mistral-7B-v0.1 MODE=ft TASK=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/BOOLQ LR=2e-4 BS=2 RANK=16 STEPS=160000 SEED=0 bash fo_lora_finetune_synthetic.sh 1>>$OUT_3 2>>$ERR_3 &

# wait

