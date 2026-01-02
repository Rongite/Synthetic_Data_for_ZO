#!/bin/bash -l

cd /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models

OUT_0=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/Copa/fo_full/original/1e-6_original.out
ERR_0=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/Copa/fo_full/original/1e-6_original.err
MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/Copa LR=1e-6 BS=16 STEPS=20000 SEED=0 bash fo_full_finetune_original.sh 1>>$OUT_0 2>>$ERR_0 & 

OUT_1=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/Copa/fo_full/original/5e-7_original.out
ERR_1=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/Copa/fo_full/original/5e-7_original.err
MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/Copa LR=5e-7 BS=16 STEPS=20000 SEED=0 bash fo_full_finetune_original.sh 1>>$OUT_1 2>>$ERR_1 &

OUT_2=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/Copa/fo_full/original/2e-7_original.out
ERR_2=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/Copa/fo_full/original/2e-7_original.err
MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/Copa LR=2e-7 BS=16 STEPS=20000 SEED=0 bash fo_full_finetune_original.sh 1>>$OUT_2 2>>$ERR_2 &

OUT_3=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/Copa/fo_full/original/1e-7_original.out
ERR_3=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/Copa/fo_full/original/1e-7_original.err
MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/Copa LR=1e-7 BS=16 STEPS=20000 SEED=0 bash fo_full_finetune_original.sh 1>>$OUT_3 2>>$ERR_3 &

wait