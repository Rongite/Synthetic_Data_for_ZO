#!/bin/bash -l

# polaris
cd /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models

# OUT_0=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/RTE/zo/original/1e-6_original.out
# ERR_0=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/RTE/zo/original/1e-6_original.err
# CUDA_VISIBLE_DEVICES=0 MODEL=mistralai/Mistral-Nemo-Base-2407 MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/RTE LR=1e-6 BS=16 STEPS=20000 SEED=0 bash mezo_finetune_original.sh 1>>$OUT_0 2>>$ERR_0

OUT_1=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/RTE/zo/original/5e-7_original.out
ERR_1=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/RTE/zo/original/5e-7_original.err
CUDA_VISIBLE_DEVICES=0 MODEL=mistralai/Mistral-Nemo-Base-2407 MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/RTE LR=5e-7 BS=16 STEPS=20000 SEED=0 bash mezo_finetune_original.sh 1>>$OUT_1 2>>$ERR_1

OUT_2=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/RTE/zo/original/2e-7_original.out
ERR_2=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/RTE/zo/original/2e-7_original.err
CUDA_VISIBLE_DEVICES=0 MODEL=mistralai/Mistral-Nemo-Base-2407 MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/RTE LR=2e-7 BS=16 STEPS=20000 SEED=0 bash mezo_finetune_original.sh 1>>$OUT_2 2>>$ERR_2

OUT_3=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/RTE/zo/original/1e-7_original.out
ERR_3=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/RTE/zo/original/1e-7_original.err
CUDA_VISIBLE_DEVICES=0 MODEL=mistralai/Mistral-Nemo-Base-2407 MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/RTE LR=1e-7 BS=16 STEPS=20000 SEED=0 bash mezo_finetune_original.sh 1>>$OUT_3 2>>$ERR_3

wait


# # gh200

# cd /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models

# OUT_0=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/RTE/llama_3.2_1b/1e-6_original.out
# ERR_0=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/RTE/llama_3.2_1b/1e-6_original.err
# MODEL=mistralai/Mistral-Nemo-Base-2407 MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/RTE LR=1e-6 BS=16 STEPS=20000 SEED=0 bash fo_full_finetune_original.sh 1>>$OUT_0 2>>$ERR_0

# OUT_1=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/RTE/llama_3.2_1b/5e-7_original.out
# ERR_1=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/RTE/llama_3.2_1b/5e-7_original.err
# MODEL=mistralai/Mistral-Nemo-Base-2407 MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/RTE LR=5e-7 BS=16 STEPS=20000 SEED=0 bash fo_full_finetune_original.sh 1>>$OUT_1 2>>$ERR_1

# wait

# OUT_2=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/RTE/llama_3.2_1b/2e-7_original.out
# ERR_2=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/RTE/llama_3.2_1b/2e-7_original.err
# MODEL=mistralai/Mistral-Nemo-Base-2407 MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/RTE LR=2e-7 BS=16 STEPS=20000 SEED=0 bash fo_full_finetune_original.sh 1>>$OUT_2 2>>$ERR_2

# OUT_3=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/RTE/llama_3.2_1b/1e-7_original.out
# ERR_3=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/RTE/llama_3.2_1b/1e-7_original.err
# MODEL=mistralai/Mistral-Nemo-Base-2407 MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/RTE LR=1e-7 BS=16 STEPS=20000 SEED=0 bash fo_full_finetune_original.sh 1>>$OUT_3 2>>$ERR_3

# wait

## siai

# cd /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models

# # independent
# OUT_0=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/RTE/llama_3.2_1b/1e-6_original.out
# ERR_0=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/RTE/llama_3.2_1b/1e-6_original.err
# MODEL=mistralai/Mistral-Nemo-Base-2407 MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/RTE LR=1e-6 BS=16 STEPS=20000 SEED=0 bash fo_full_finetune_original.sh 1>>$OUT_0 2>>$ERR_0

# # independent
# OUT_1=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/RTE/llama_3.2_1b/5e-7_original.out
# ERR_1=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/RTE/llama_3.2_1b/5e-7_original.err
# MODEL=mistralai/Mistral-Nemo-Base-2407 MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/RTE LR=5e-7 BS=16 STEPS=20000 SEED=0 bash fo_full_finetune_original.sh 1>>$OUT_1 2>>$ERR_1

# OUT_2=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/RTE/llama_3.2_1b/2e-7_original.out
# ERR_2=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/RTE/llama_3.2_1b/2e-7_original.err
# MODEL=mistralai/Mistral-Nemo-Base-2407 MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/RTE LR=2e-7 BS=16 STEPS=20000 SEED=0 bash fo_full_finetune_original.sh 1>>$OUT_2 2>>$ERR_2

# OUT_3=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/RTE/llama_3.2_1b/1e-7_original.out
# ERR_3=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/RTE/llama_3.2_1b/1e-7_original.err
# MODEL=mistralai/Mistral-Nemo-Base-2407 MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/RTE LR=1e-7 BS=16 STEPS=20000 SEED=0 bash fo_full_finetune_original.sh 1>>$OUT_3 2>>$ERR_3

# wait