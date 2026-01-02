#!/bin/bash -l

cd /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models

OUT_0=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/CB/fo_lora/rejection_sampling/1e-4_lora_rk8.out
ERR_0=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/CB/fo_lora/rejection_sampling/1e-4_lora_rk8.err
MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/CB LR=1e-4 BS=16 RANK=8 STEPS=20000 SEED=0 bash fo_lora_finetune_synthetic.sh 1>>$OUT_0 2>>$ERR_0 &

OUT_2=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/CB/fo_lora/rejection_sampling/1e-4_lora_rk16.out
ERR_2=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/CB/fo_lora/rejection_sampling/1e-4_lora_rk16.err
MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/CB LR=1e-4 BS=16 RANK=16 STEPS=20000 SEED=0 bash fo_lora_finetune_synthetic.sh 1>>$OUT_2 2>>$ERR_2 &

wait

OUT_1=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/CB/fo_lora/rejection_sampling/2e-4_lora_rk8.out
ERR_1=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/CB/fo_lora/rejection_sampling/2e-4_lora_rk8.err
MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/CB LR=2e-4 BS=16 RANK=8 STEPS=20000 SEED=0 bash fo_lora_finetune_synthetic.sh 1>>$OUT_1 2>>$ERR_1 &

OUT_3=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/CB/fo_lora/rejection_sampling/2e-4_lora_rk16.out
ERR_3=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/CB/fo_lora/rejection_sampling/2e-4_lora_rk16.err
MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/CB LR=2e-4 BS=16 RANK=16 STEPS=20000 SEED=0 bash fo_lora_finetune_synthetic.sh 1>>$OUT_3 2>>$ERR_3 &

wait
