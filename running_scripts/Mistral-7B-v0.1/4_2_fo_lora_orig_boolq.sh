#!/bin/bash -l

# polaris 60h
cd /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models

OUT_0=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/BOOLQ/fo_lora/original/1e-4_lora_rk8.out
ERR_0=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/BOOLQ/fo_lora/original/1e-4_lora_rk8.err
MODEL=mistralai/Mistral-7B-v0.1 MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/BOOLQ LR=1e-4 BS=2 RANK=8 STEPS=160000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_0 2>>$ERR_0 &

OUT_1=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/BOOLQ/fo_lora/original/2e-4_lora_rk8.out
ERR_1=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/BOOLQ/fo_lora/original/2e-4_lora_rk8.err
MODEL=mistralai/Mistral-7B-v0.1 MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/BOOLQ LR=2e-4 BS=2 RANK=8 STEPS=160000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_1 2>>$ERR_1 &

wait

OUT_2=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/BOOLQ/fo_lora/original/1e-4_lora_rk16.out
ERR_2=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/BOOLQ/fo_lora/original/1e-4_lora_rk16.err
MODEL=mistralai/Mistral-7B-v0.1 MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/BOOLQ LR=1e-4 BS=2 RANK=16 STEPS=160000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_2 2>>$ERR_2 &

OUT_3=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/BOOLQ/fo_lora/original/2e-4_lora_rk16.out
ERR_3=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/BOOLQ/fo_lora/original/2e-4_lora_rk16.err
MODEL=mistralai/Mistral-7B-v0.1 MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/BOOLQ LR=2e-4 BS=2 RANK=16 STEPS=160000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_3 2>>$ERR_3 &

wait

# siai
# cd /home/jlong1/Downloads/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models

# # indenpendence
# OUT_0=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/BOOLQ/llama_3.2_1b/1e-4_lora_rk8.out
# ERR_0=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/BOOLQ/llama_3.2_1b/1e-4_lora_rk8.err
# MODEL=mistralai/Mistral-7B-v0.1 MODE=ft TASK=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/BOOLQ LR=1e-4 BS=2 RANK=8 STEPS=160000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_0 2>>$ERR_0 

# # indenpendence
# OUT_1=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/BOOLQ/llama_3.2_1b/2e-4_lora_rk8.out
# ERR_1=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/BOOLQ/llama_3.2_1b/2e-4_lora_rk8.err
# MODEL=mistralai/Mistral-7B-v0.1 MODE=ft TASK=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/BOOLQ LR=2e-4 BS=2 RANK=8 STEPS=160000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_1 2>>$ERR_1

# # indenpendence
# OUT_2=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/BOOLQ/llama_3.2_1b/1e-4_lora_rk16.out
# ERR_2=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/BOOLQ/llama_3.2_1b/1e-4_lora_rk16.err
# MODEL=mistralai/Mistral-7B-v0.1 MODE=ft TASK=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/BOOLQ LR=1e-4 BS=2 RANK=16 STEPS=160000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_2 2>>$ERR_2

# # indenpendence
# OUT_3=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/BOOLQ/llama_3.2_1b/2e-4_lora_rk16.out
# ERR_3=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/BOOLQ/llama_3.2_1b/2e-4_lora_rk16.err
# MODEL=mistralai/Mistral-7B-v0.1 MODE=ft TASK=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/BOOLQ LR=2e-4 BS=2 RANK=16 STEPS=160000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_3 2>>$ERR_3
