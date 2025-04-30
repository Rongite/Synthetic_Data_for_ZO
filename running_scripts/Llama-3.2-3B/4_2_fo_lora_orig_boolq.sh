#!/bin/bash -l

# polaris 45h
cd /grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models

OUT_0=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Llama-3.2-3B/BOOLQ/fo_lora/original/1e-4_lora_rk8.out
ERR_0=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Llama-3.2-3B/BOOLQ/fo_lora/original/1e-4_lora_rk8.err
CUDA_VISIBLE_DEVICES=0,1 MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/original/BOOLQ LR=1e-4 BS=4 RANK=8 STEPS=80000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_0 2>>$ERR_0 &

OUT_1=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Llama-3.2-3B/BOOLQ/fo_lora/original/2e-4_lora_rk8.out
ERR_1=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Llama-3.2-3B/BOOLQ/fo_lora/original/2e-4_lora_rk8.err
CUDA_VISIBLE_DEVICES=2,3 MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/original/BOOLQ LR=2e-4 BS=4 RANK=8 STEPS=80000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_1 2>>$ERR_1 &

wait

OUT_2=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Llama-3.2-3B/BOOLQ/fo_lora/original/1e-4_lora_rk16.out
ERR_2=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Llama-3.2-3B/BOOLQ/fo_lora/original/1e-4_lora_rk16.err
CUDA_VISIBLE_DEVICES=0,1 MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/original/BOOLQ LR=1e-4 BS=4 RANK=16 STEPS=80000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_2 2>>$ERR_2 &

OUT_3=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Llama-3.2-3B/BOOLQ/fo_lora/original/2e-4_lora_rk16.out
ERR_3=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Llama-3.2-3B/BOOLQ/fo_lora/original/2e-4_lora_rk16.err
CUDA_VISIBLE_DEVICES=2,3 MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/original/BOOLQ LR=2e-4 BS=4 RANK=16 STEPS=80000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_3 2>>$ERR_3 &

wait

OUT_4=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Llama-3.2-3B/BOOLQ/fo_lora/original/1e-4_lora_rk32.out
ERR_4=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Llama-3.2-3B/BOOLQ/fo_lora/original/1e-4_lora_rk32.err
CUDA_VISIBLE_DEVICES=0,1 MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/original/BOOLQ LR=1e-4 BS=4 RANK=32 STEPS=80000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_4 2>>$ERR_4 &

OUT_5=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Llama-3.2-3B/BOOLQ/fo_lora/original/2e-4_lora_rk32.out
ERR_5=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Llama-3.2-3B/BOOLQ/fo_lora/original/2e-4_lora_rk32.err
CUDA_VISIBLE_DEVICES=2,3 MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/original/BOOLQ LR=2e-4 BS=4 RANK=32 STEPS=80000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_5 2>>$ERR_5 &

wait


# siai
# cd /home/jlong1/Downloads/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models

# # indenpendence
# OUT_0=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/BOOLQ/llama_3.2_1b/1e-4_lora_rk8.out
# ERR_0=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/BOOLQ/llama_3.2_1b/1e-4_lora_rk8.err
# MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/BOOLQ LR=1e-4 BS=4 RANK=8 STEPS=80000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_0 2>>$ERR_0 

# # indenpendence
# OUT_1=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/BOOLQ/llama_3.2_1b/2e-4_lora_rk8.out
# ERR_1=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/BOOLQ/llama_3.2_1b/2e-4_lora_rk8.err
# MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/BOOLQ LR=2e-4 BS=4 RANK=8 STEPS=80000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_1 2>>$ERR_1

# # indenpendence
# OUT_2=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/BOOLQ/llama_3.2_1b/1e-4_lora_rk16.out
# ERR_2=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/BOOLQ/llama_3.2_1b/1e-4_lora_rk16.err
# MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/BOOLQ LR=1e-4 BS=4 RANK=16 STEPS=80000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_2 2>>$ERR_2

# # indenpendence
# OUT_3=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/BOOLQ/llama_3.2_1b/2e-4_lora_rk16.out
# ERR_3=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/BOOLQ/llama_3.2_1b/2e-4_lora_rk16.err
# MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/BOOLQ LR=2e-4 BS=4 RANK=16 STEPS=80000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_3 2>>$ERR_3
