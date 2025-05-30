#!/bin/bash -l

# cd /grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models
# module use /soft/modulefiles
# module load conda
# module load cudatoolkit-standalone/11.8.0
# export CUDA_HOME=/soft/compilers/cudatoolkit/cuda-11.8.0
# conda activate mezo

# export TRANSFORMERS_CACHE=/grand/sbi-fair/jikaiLoong/models/transformers_cache

cd /home/jlong1/Downloads/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models

OUT_0=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/Copa/llama_3.2_1b_instruct/1e-6_synthetic.out
ERR_0=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/Copa/llama_3.2_1b_instruct/1e-6_synthetic.err
CUDA_VISIBLE_DEVICES=0 MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/Copa LR=1e-4 BS=16 RANK=8 STEPS=20000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_0 2>>$ERR_0 & 

OUT_1=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/Copa/llama_3.2_1b_instruct/5e-7_synthetic.out
ERR_1=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/Copa/llama_3.2_1b_instruct/5e-7_synthetic.err
CUDA_VISIBLE_DEVICES=0 MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/Copa LR=2e-4 BS=16 RANK=8 STEPS=20000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_1 2>>$ERR_1 &

OUT_2=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/Copa/llama_3.2_1b_instruct/2e-7_synthetic.out
ERR_2=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/Copa/llama_3.2_1b_instruct/2e-7_synthetic.err
CUDA_VISIBLE_DEVICES=1 MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/Copa LR=1e-4 BS=16 RANK=16 STEPS=20000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_2 2>>$ERR_2 &

OUT_3=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/Copa/llama_3.2_1b_instruct/1e-7_synthetic.out
ERR_3=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/Copa/llama_3.2_1b_instruct/1e-7_synthetic.err
CUDA_VISIBLE_DEVICES=1 MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/Copa LR=2e-4 BS=16 RANK=16 STEPS=20000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_3 2>>$ERR_3 &
