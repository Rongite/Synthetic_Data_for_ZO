#!/bin/bash -l

# polaris 6h
cd /grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models

OUT_0=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/rejection_sampling/2_first_order_full/result/RTE/llama_3.2_1b/1e-6_synthetic.out
ERR_0=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/rejection_sampling/2_first_order_full/result/RTE/llama_3.2_1b/1e-6_synthetic.err
CUDA_VISIBLE_DEVICES=0 MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/RTE LR=1e-6 BS=16 STEPS=20000 SEED=0 bash fo_full_finetune_synthetic.sh 1>>$OUT_0 2>>$ERR_0 & 

OUT_2=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/rejection_sampling/2_first_order_full/result/RTE/llama_3.2_1b/2e-7_synthetic.out
ERR_2=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/rejection_sampling/2_first_order_full/result/RTE/llama_3.2_1b/2e-7_synthetic.err
CUDA_VISIBLE_DEVICES=1 MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/RTE LR=2e-7 BS=16 STEPS=20000 SEED=0 bash fo_full_finetune_synthetic.sh 1>>$OUT_2 2>>$ERR_2 &

OUT_1=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/rejection_sampling/2_first_order_full/result/RTE/llama_3.2_1b/5e-7_synthetic.out
ERR_1=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/rejection_sampling/2_first_order_full/result/RTE/llama_3.2_1b/5e-7_synthetic.err
CUDA_VISIBLE_DEVICES=2 MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/RTE LR=5e-7 BS=16 STEPS=20000 SEED=0 bash fo_full_finetune_synthetic.sh 1>>$OUT_1 2>>$ERR_1 &

OUT_3=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/rejection_sampling/2_first_order_full/result/RTE/llama_3.2_1b/1e-7_synthetic.out
ERR_3=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/rejection_sampling/2_first_order_full/result/RTE/llama_3.2_1b/1e-7_synthetic.err
CUDA_VISIBLE_DEVICES=3 MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/RTE LR=1e-7 BS=16 STEPS=20000 SEED=0 bash fo_full_finetune_synthetic.sh 1>>$OUT_3 2>>$ERR_3 &

wait

# siai
# cd /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models

# OUT_0=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/2_first_order_full/result/RTE/llama_3.2_1b/1e-6_synthetic.out
# ERR_0=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/2_first_order_full/result/RTE/llama_3.2_1b/1e-6_synthetic.err
# CUDA_VISIBLE_DEVICES=0 MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/RTE LR=1e-6 BS=16 STEPS=20000 SEED=0 bash fo_full_finetune_synthetic.sh 1>>$OUT_0 2>>$ERR_0 & 

# OUT_2=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/2_first_order_full/result/RTE/llama_3.2_1b/2e-7_synthetic.out
# ERR_2=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/2_first_order_full/result/RTE/llama_3.2_1b/2e-7_synthetic.err
# CUDA_VISIBLE_DEVICES=1 MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/RTE LR=2e-7 BS=16 STEPS=20000 SEED=0 bash fo_full_finetune_synthetic.sh 1>>$OUT_2 2>>$ERR_2 &

# wait

# OUT_1=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/2_first_order_full/result/RTE/llama_3.2_1b/5e-7_synthetic.out
# ERR_1=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/2_first_order_full/result/RTE/llama_3.2_1b/5e-7_synthetic.err
# CUDA_VISIBLE_DEVICES=0 MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/RTE LR=5e-7 BS=16 STEPS=20000 SEED=0 bash fo_full_finetune_synthetic.sh 1>>$OUT_1 2>>$ERR_1 &

# OUT_3=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/2_first_order_full/result/RTE/llama_3.2_1b/1e-7_synthetic.out
# ERR_3=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/2_first_order_full/result/RTE/llama_3.2_1b/1e-7_synthetic.err
# CUDA_VISIBLE_DEVICES=1 MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/RTE LR=1e-7 BS=16 STEPS=20000 SEED=0 bash fo_full_finetune_synthetic.sh 1>>$OUT_3 2>>$ERR_3 &

# wait