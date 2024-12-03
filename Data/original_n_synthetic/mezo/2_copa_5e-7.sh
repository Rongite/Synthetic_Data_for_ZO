#!/bin/bash -l

# conda activate mezo
cd /home/jlong1/Downloads/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models

OUT_1=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original_n_synthetic/mezo/result/Copa/5e-7_original_n_synthetic.out
ERR_1=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original_n_synthetic/mezo/result/Copa/5e-7_original_n_synthetic.err
CUDA_VISIBLE_DEVICES=1 MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original_n_synthetic/mezo/Copa LR=5e-7 BS=16 EPS=1e-3 STEPS=20000 SEED=0 bash mezo_finetune_synthetic.sh 1>>$OUT_1 2>>$ERR_1 &
