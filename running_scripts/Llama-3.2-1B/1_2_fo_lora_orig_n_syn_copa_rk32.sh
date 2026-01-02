#!/bin/bash -l

cd /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models

OUT_0=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/Copa/fo_lora/original/1e-4_lora_rk32.out
ERR_0=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/Copa/fo_lora/original/1e-4_lora_rk32.err
MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/Copa LR=1e-4 BS=16 RANK=32 STEPS=20000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_0 2>>$ERR_0 &

OUT_1=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/Copa/fo_lora/original/2e-4_lora_rk32.out
ERR_1=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/Copa/fo_lora/original/2e-4_lora_rk32.err
MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/Copa LR=2e-4 BS=16 RANK=32 STEPS=20000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_1 2>>$ERR_1 &

OUT_2=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/Copa/fo_lora/rejection_sampling/1e-4_lora_rk32.out
ERR_2=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/Copa/fo_lora/rejection_sampling/1e-4_lora_rk32.err
MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/Copa LR=1e-4 BS=16 RANK=32 STEPS=20000 SEED=0 bash fo_lora_finetune_synthetic.sh 1>>$OUT_2 2>>$ERR_2 &

OUT_3=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/Copa/fo_lora/rejection_sampling/2e-4_lora_rk32.out
ERR_3=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/Copa/fo_lora/rejection_sampling/2e-4_lora_rk32.err
MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/Copa LR=2e-4 BS=16 RANK=32 STEPS=20000 SEED=0 bash fo_lora_finetune_synthetic.sh 1>>$OUT_3 2>>$ERR_3 &

wait
