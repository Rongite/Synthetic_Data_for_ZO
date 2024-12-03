#!/bin/bash -l

# conda activate mezo
cd /home/jlong1/Downloads/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models

OUT_0=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/rejection_sampling/mezo/result/Copa/original/1e-6_original.out
ERR_0=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/rejection_sampling/mezo/result/Copa/original/1e-6_original.err
CUDA_VISIBLE_DEVICES=0 MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/Copa LR=1e-6 BS=16 EPS=1e-3 STEPS=20000 SEED=0 bash mezo_finetune_original.sh 1>>$OUT_0 2>>$ERR_0 & 

OUT_1=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/rejection_sampling/mezo/result/Copa/rejection_sampling/1e-6_rejection-sampling.out
ERR_1=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/rejection_sampling/mezo/result/Copa/rejection_sampling/1e-6_rejection-sampling.err
CUDA_VISIBLE_DEVICES=1 MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/rejection_sampling/mezo/Copa LR=1e-6 BS=16 EPS=1e-3 STEPS=20000 SEED=0 bash mezo_finetune_synthetic.sh 1>>$OUT_1 2>>$ERR_1 &
