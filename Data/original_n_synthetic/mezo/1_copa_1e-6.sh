#!/bin/bash -l

# conda activate mezo
cd /grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models
# module use /soft/modulefiles
# module load conda
# module load cudatoolkit-standalone/11.8.0
# export CUDA_HOME=/soft/compilers/cudatoolkit/cuda-11.8.0
# conda activate mezo

export TRANSFORMERS_CACHE=/grand/sbi-fair/jikaiLoong/models/transformers_cache

OUT_0=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/original_n_synthetic/mezo/result/Copa/1e-6_original_n_synthetic.out
ERR_0=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/original_n_synthetic/mezo/result/Copa/1e-6_original_n_synthetic.err
CUDA_VISIBLE_DEVICES=0 MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/original_n_synthetic/mezo/Copa LR=1e-6 BS=16 EPS=1e-3 STEPS=20000 SEED=0 bash mezo_finetune_orig_n_syn.sh 1>>$OUT_0 2>>$ERR_0 & 
