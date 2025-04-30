#!/bin/bash -l

# polaris 10h
cd /grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models

OUT_0=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/ArcC_MC/zo/original/1e-6_original.out
ERR_0=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/ArcC_MC/zo/original/1e-6_original.err
CUDA_VISIBLE_DEVICES=0 MODEL=meta-llama/Mistral-7B-v0.1 MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/original/ArcC_MC LR=1e-6 BS=4 STEPS=80000 SEED=0 bash mezo_finetune_original.sh 1>>$OUT_0 2>>$ERR_0 &

OUT_1=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/ArcC_MC/zo/original/5e-7_original.out
ERR_1=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/ArcC_MC/zo/original/5e-7_original.err
CUDA_VISIBLE_DEVICES=1 MODEL=meta-llama/Mistral-7B-v0.1 MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/original/ArcC_MC LR=5e-7 BS=4 STEPS=80000 SEED=0 bash mezo_finetune_original.sh 1>>$OUT_1 2>>$ERR_1 &

OUT_2=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/ArcC_MC/zo/original/2e-7_original.out
ERR_2=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/ArcC_MC/zo/original/2e-7_original.err
CUDA_VISIBLE_DEVICES=2 MODEL=meta-llama/Mistral-7B-v0.1 MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/original/ArcC_MC LR=2e-7 BS=4 STEPS=80000 SEED=0 bash mezo_finetune_original.sh 1>>$OUT_2 2>>$ERR_2 &

OUT_3=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/ArcC_MC/zo/original/1e-7_original.out
ERR_3=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/ArcC_MC/zo/original/1e-7_original.err
CUDA_VISIBLE_DEVICES=3 MODEL=meta-llama/Mistral-7B-v0.1 MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/original/ArcC_MC LR=1e-7 BS=4 STEPS=80000 SEED=0 bash mezo_finetune_original.sh 1>>$OUT_3 2>>$ERR_3 &

wait

# # gh200

# cd /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models

# OUT_0=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/ArcC_MC/llama_3.2_1b/1e-6_original.out
# ERR_0=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/ArcC_MC/llama_3.2_1b/1e-6_original.err
# MODEL=meta-llama/Mistral-7B-v0.1 MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/ArcC_MC LR=1e-6 BS=4 STEPS=80000 SEED=0 bash fo_full_finetune_original.sh 1>>$OUT_0 2>>$ERR_0 &

# OUT_1=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/ArcC_MC/llama_3.2_1b/5e-7_original.out
# ERR_1=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/ArcC_MC/llama_3.2_1b/5e-7_original.err
# MODEL=meta-llama/Mistral-7B-v0.1 MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/ArcC_MC LR=5e-7 BS=4 STEPS=80000 SEED=0 bash fo_full_finetune_original.sh 1>>$OUT_1 2>>$ERR_1 &

# wait

# OUT_2=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/ArcC_MC/llama_3.2_1b/2e-7_original.out
# ERR_2=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/ArcC_MC/llama_3.2_1b/2e-7_original.err
# MODEL=meta-llama/Mistral-7B-v0.1 MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/ArcC_MC LR=2e-7 BS=4 STEPS=80000 SEED=0 bash fo_full_finetune_original.sh 1>>$OUT_2 2>>$ERR_2 &

# OUT_3=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/ArcC_MC/llama_3.2_1b/1e-7_original.out
# ERR_3=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/ArcC_MC/llama_3.2_1b/1e-7_original.err
# MODEL=meta-llama/Mistral-7B-v0.1 MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/ArcC_MC LR=1e-7 BS=4 STEPS=80000 SEED=0 bash fo_full_finetune_original.sh 1>>$OUT_3 2>>$ERR_3 &

# wait

## siai

# cd /home/jlong1/Downloads/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models

# # independent
# OUT_0=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/ArcC_MC/llama_3.2_1b/1e-6_original.out
# ERR_0=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/ArcC_MC/llama_3.2_1b/1e-6_original.err
# MODEL=meta-llama/Mistral-7B-v0.1 MODE=ft TASK=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/ArcC_MC LR=1e-6 BS=4 STEPS=80000 SEED=0 bash fo_full_finetune_original.sh 1>>$OUT_0 2>>$ERR_0 &

# # independent
# OUT_1=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/ArcC_MC/llama_3.2_1b/5e-7_original.out
# ERR_1=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/ArcC_MC/llama_3.2_1b/5e-7_original.err
# MODEL=meta-llama/Mistral-7B-v0.1 MODE=ft TASK=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/ArcC_MC LR=5e-7 BS=4 STEPS=80000 SEED=0 bash fo_full_finetune_original.sh 1>>$OUT_1 2>>$ERR_1

# OUT_2=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/ArcC_MC/llama_3.2_1b/2e-7_original.out
# ERR_2=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/ArcC_MC/llama_3.2_1b/2e-7_original.err
# MODEL=meta-llama/Mistral-7B-v0.1 MODE=ft TASK=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/ArcC_MC LR=2e-7 BS=4 STEPS=80000 SEED=0 bash fo_full_finetune_original.sh 1>>$OUT_2 2>>$ERR_2 &

# OUT_3=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/ArcC_MC/llama_3.2_1b/1e-7_original.out
# ERR_3=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/2_first_order_full/result/ArcC_MC/llama_3.2_1b/1e-7_original.err
# MODEL=meta-llama/Mistral-7B-v0.1 MODE=ft TASK=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/ArcC_MC LR=1e-7 BS=4 STEPS=80000 SEED=0 bash fo_full_finetune_original.sh 1>>$OUT_3 2>>$ERR_3 &

# wait