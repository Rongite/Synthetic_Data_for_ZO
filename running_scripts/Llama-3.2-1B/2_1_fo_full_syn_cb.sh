#!/bin/bash -l

cd /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models

OUT_0=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/CB/fo_full/rejection_sampling/1e-6_synthetic.out
ERR_0=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/CB/fo_full/rejection_sampling/1e-6_synthetic.err
MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/CB LR=1e-6 BS=16 STEPS=20000 SEED=0 bash fo_full_finetune_synthetic.sh 1>>$OUT_0 2>>$ERR_0 & 

OUT_2=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/CB/fo_full/rejection_sampling/2e-7_synthetic.out
ERR_2=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/CB/fo_full/rejection_sampling/2e-7_synthetic.err
MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/CB LR=2e-7 BS=16 STEPS=20000 SEED=0 bash fo_full_finetune_synthetic.sh 1>>$OUT_2 2>>$ERR_2 &

wait

OUT_1=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/CB/fo_full/rejection_sampling/5e-7_synthetic.out
ERR_1=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/CB/fo_full/rejection_sampling/5e-7_synthetic.err
MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/CB LR=5e-7 BS=16 STEPS=20000 SEED=0 bash fo_full_finetune_synthetic.sh 1>>$OUT_1 2>>$ERR_1 &

OUT_3=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/CB/fo_full/rejection_sampling/1e-7_synthetic.out
ERR_3=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-1B/CB/fo_full/rejection_sampling/1e-7_synthetic.err
MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/CB LR=1e-7 BS=16 STEPS=20000 SEED=0 bash fo_full_finetune_synthetic.sh 1>>$OUT_3 2>>$ERR_3 &

wait