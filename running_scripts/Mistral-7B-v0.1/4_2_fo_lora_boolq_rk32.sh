#!/bin/bash -l

# polaris 60h
cd /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models

OUT_1=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/BOOLQ/fo_lora/original/1e-4_lora_rk32.out
ERR_1=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/BOOLQ/fo_lora/original/1e-4_lora_rk32.err
MODEL=mistralai/Mistral-7B-v0.1 MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/BOOLQ LR=1e-4 BS=2 RANK=32 STEPS=160000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_1 2>>$ERR_1 &

wait

OUT_2=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/BOOLQ/fo_lora/original/2e-4_lora_rk32.out
ERR_2=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/BOOLQ/fo_lora/original/2e-4_lora_rk32.err
MODEL=mistralai/Mistral-7B-v0.1 MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/BOOLQ LR=2e-4 BS=2 RANK=32 STEPS=160000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_2 2>>$ERR_2 &

wait

# modify bs
OUT_3=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/BOOLQ/fo_lora/rejection_sampling/1e-4_lora_rk32.out
ERR_3=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/BOOLQ/fo_lora/rejection_sampling/1e-4_lora_rk32.err
MODEL=mistralai/Mistral-7B-v0.1 MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/BOOLQ LR=1e-4 BS=4 RANK=32 STEPS=80000 SEED=0 bash fo_lora_finetune_synthetic.sh 1>>$OUT_3 2>>$ERR_3 &

wait

OUT_4=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/BOOLQ/fo_lora/rejection_sampling/2e-4_lora_rk32.out
ERR_4=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/BOOLQ/fo_lora/rejection_sampling/2e-4_lora_rk32.err
MODEL=mistralai/Mistral-7B-v0.1 MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/BOOLQ LR=2e-4 BS=2 RANK=32 STEPS=160000 SEED=0 bash fo_lora_finetune_synthetic.sh 1>>$OUT_4 2>>$ERR_4 &

wait

