#!/bin/bash -l

cd /grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models
module use /soft/modulefiles
module load conda
module load cudatoolkit-standalone/11.8.0
export CUDA_HOME=/soft/compilers/cudatoolkit/cuda-11.8.0
conda activate mezo

export TRANSFORMERS_CACHE=/grand/sbi-fair/jikaiLoong/models/transformers_cache

OUT_0=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/rejection_sampling/mezo/result/Copa/llama_3.2_3b/1e-6_synthetic.out
ERR_0=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/rejection_sampling/mezo/result/Copa/llama_3.2_3b/1e-6_synthetic.err
CUDA_VISIBLE_DEVICES=0 MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/rejection_sampling/mezo/Copa LR=1e-6 BS=16 EPS=1e-3 STEPS=20000 SEED=0 bash mezo_finetune_synthetic.sh 1>>$OUT_0 2>>$ERR_0 & 

OUT_1=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/rejection_sampling/mezo/result/Copa/llama_3.2_3b/5e-7_synthetic.out
ERR_1=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/rejection_sampling/mezo/result/Copa/llama_3.2_3b/5e-7_synthetic.err
CUDA_VISIBLE_DEVICES=1 MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/rejection_sampling/mezo/Copa LR=5e-7 BS=16 EPS=1e-3 STEPS=20000 SEED=0 bash mezo_finetune_synthetic.sh 1>>$OUT_1 2>>$ERR_1 &

OUT_2=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/rejection_sampling/mezo/result/Copa/llama_3.2_3b/2e-7_synthetic.out
ERR_2=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/rejection_sampling/mezo/result/Copa/llama_3.2_3b/2e-7_synthetic.err
CUDA_VISIBLE_DEVICES=2 MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/rejection_sampling/mezo/Copa LR=2e-7 BS=16 EPS=1e-3 STEPS=20000 SEED=0 bash mezo_finetune_synthetic.sh 1>>$OUT_2 2>>$ERR_2 &

OUT_3=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/rejection_sampling/mezo/result/Copa/llama_3.2_3b/1e-7_synthetic.out
ERR_3=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/rejection_sampling/mezo/result/Copa/llama_3.2_3b/1e-7_synthetic.err
CUDA_VISIBLE_DEVICES=3 MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/rejection_sampling/mezo/Copa LR=1e-7 BS=16 EPS=1e-3 STEPS=20000 SEED=0 bash mezo_finetune_synthetic.sh 1>>$OUT_3 2>>$ERR_3 &
