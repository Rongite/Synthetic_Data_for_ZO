#!/bin/bash -l

# polaris 5h
cd /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models

OUT_0=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3-70B/BOOLQ/zo/rejection_sampling/1e-6_synthetic.out
ERR_0=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3-70B/BOOLQ/zo/rejection_sampling/1e-6_synthetic.err
CUDA_VISIBLE_DEVICES=0 MODEL=meta-llama/Meta-Llama-3-70B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/BOOLQ LR=1e-6 BS=4 STEPS=80000 SEED=0 bash mezo_finetune_synthetic.sh 1>>$OUT_0 2>>$ERR_0

OUT_1=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3-70B/BOOLQ/zo/rejection_sampling/5e-7_synthetic.out
ERR_1=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3-70B/BOOLQ/zo/rejection_sampling/5e-7_synthetic.err
CUDA_VISIBLE_DEVICES=0 MODEL=meta-llama/Meta-Llama-3-70B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/BOOLQ LR=5e-7 BS=4 STEPS=80000 SEED=0 bash mezo_finetune_synthetic.sh 1>>$OUT_1 2>>$ERR_1

OUT_2=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3-70B/BOOLQ/zo/rejection_sampling/2e-7_synthetic.out
ERR_2=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3-70B/BOOLQ/zo/rejection_sampling/2e-7_synthetic.err
CUDA_VISIBLE_DEVICES=0 MODEL=meta-llama/Meta-Llama-3-70B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/BOOLQ LR=2e-7 BS=4 STEPS=80000 SEED=0 bash mezo_finetune_synthetic.sh 1>>$OUT_2 2>>$ERR_2

OUT_3=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3-70B/BOOLQ/zo/rejection_sampling/1e-7_synthetic.out
ERR_3=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3-70B/BOOLQ/zo/rejection_sampling/1e-7_synthetic.err
CUDA_VISIBLE_DEVICES=0 MODEL=meta-llama/Meta-Llama-3-70B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/BOOLQ LR=1e-7 BS=4 STEPS=80000 SEED=0 bash mezo_finetune_synthetic.sh 1>>$OUT_3 2>>$ERR_3

wait


# # gh200

# cd /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models

# OUT_0=/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/BOOLQ/llama_3.2_1b/1e-6_original.out
# ERR_0=/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/BOOLQ/llama_3.2_1b/1e-6_original.err
# MODEL=meta-llama/Meta-Llama-3-70B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/original/BOOLQ LR=1e-6 BS=4 STEPS=80000 SEED=0 bash fo_full_finetune_original.sh 1>>$OUT_0 2>>$ERR_0

# OUT_1=/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/BOOLQ/llama_3.2_1b/5e-7_original.out
# ERR_1=/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/BOOLQ/llama_3.2_1b/5e-7_original.err
# MODEL=meta-llama/Meta-Llama-3-70B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/original/BOOLQ LR=5e-7 BS=4 STEPS=80000 SEED=0 bash fo_full_finetune_original.sh 1>>$OUT_1 2>>$ERR_1

# wait

# OUT_2=/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/BOOLQ/llama_3.2_1b/2e-7_original.out
# ERR_2=/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/BOOLQ/llama_3.2_1b/2e-7_original.err
# MODEL=meta-llama/Meta-Llama-3-70B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/original/BOOLQ LR=2e-7 BS=4 STEPS=80000 SEED=0 bash fo_full_finetune_original.sh 1>>$OUT_2 2>>$ERR_2

# OUT_3=/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/BOOLQ/llama_3.2_1b/1e-7_original.out
# ERR_3=/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/BOOLQ/llama_3.2_1b/1e-7_original.err
# MODEL=meta-llama/Meta-Llama-3-70B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/original/BOOLQ LR=1e-7 BS=4 STEPS=80000 SEED=0 bash fo_full_finetune_original.sh 1>>$OUT_3 2>>$ERR_3

# wait

## siai

# cd /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models

# # independent
# OUT_0=/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/BOOLQ/llama_3.2_1b/1e-6_original.out
# ERR_0=/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/BOOLQ/llama_3.2_1b/1e-6_original.err
# MODEL=meta-llama/Meta-Llama-3-70B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/original/BOOLQ LR=1e-6 BS=4 STEPS=80000 SEED=0 bash fo_full_finetune_original.sh 1>>$OUT_0 2>>$ERR_0

# # independent
# OUT_1=/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/BOOLQ/llama_3.2_1b/5e-7_original.out
# ERR_1=/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/BOOLQ/llama_3.2_1b/5e-7_original.err
# MODEL=meta-llama/Meta-Llama-3-70B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/original/BOOLQ LR=5e-7 BS=4 STEPS=80000 SEED=0 bash fo_full_finetune_original.sh 1>>$OUT_1 2>>$ERR_1

# OUT_2=/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/BOOLQ/llama_3.2_1b/2e-7_original.out
# ERR_2=/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/BOOLQ/llama_3.2_1b/2e-7_original.err
# MODEL=meta-llama/Meta-Llama-3-70B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/original/BOOLQ LR=2e-7 BS=4 STEPS=80000 SEED=0 bash fo_full_finetune_original.sh 1>>$OUT_2 2>>$ERR_2

# OUT_3=/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/BOOLQ/llama_3.2_1b/1e-7_original.out
# ERR_3=/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/BOOLQ/llama_3.2_1b/1e-7_original.err
# MODEL=meta-llama/Meta-Llama-3-70B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/original/BOOLQ LR=1e-7 BS=4 STEPS=80000 SEED=0 bash fo_full_finetune_original.sh 1>>$OUT_3 2>>$ERR_3

# wait