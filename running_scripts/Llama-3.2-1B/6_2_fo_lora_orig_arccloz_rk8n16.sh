#!/bin/bash -l

# polaris 6h
cd /grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models

OUT_0=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/ArcC_Cloze/llama_3.2_1b/1e-4_lora_rk8.out
ERR_0=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/ArcC_Cloze/llama_3.2_1b/1e-4_lora_rk8.err
CUDA_VISIBLE_DEVICES=0 MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/original/ArcC_Cloze LR=1e-4 BS=16 RANK=8 STEPS=20000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_0 2>>$ERR_0 &

OUT_1=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/ArcC_Cloze/llama_3.2_1b/2e-4_lora_rk8.out
ERR_1=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/ArcC_Cloze/llama_3.2_1b/2e-4_lora_rk8.err
CUDA_VISIBLE_DEVICES=1 MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/original/ArcC_Cloze LR=2e-4 BS=16 RANK=8 STEPS=20000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_1 2>>$ERR_1 &

OUT_2=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/ArcC_Cloze/llama_3.2_1b/1e-4_lora_rk16.out
ERR_2=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/ArcC_Cloze/llama_3.2_1b/1e-4_lora_rk16.err
CUDA_VISIBLE_DEVICES=2 MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/original/ArcC_Cloze LR=1e-4 BS=8 RANK=16 STEPS=40000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_2 2>>$ERR_2 &

OUT_3=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/ArcC_Cloze/llama_3.2_1b/2e-4_lora_rk16.out
ERR_3=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/ArcC_Cloze/llama_3.2_1b/2e-4_lora_rk16.err
CUDA_VISIBLE_DEVICES=3 MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/original/ArcC_Cloze LR=2e-4 BS=8 RANK=16 STEPS=40000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_3 2>>$ERR_3 &

wait

# siai
# cd /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models

# # indenpendence
# OUT_0=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/ArcC_Cloze/llama_3.2_1b/1e-4_lora_rk8.out
# ERR_0=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/ArcC_Cloze/llama_3.2_1b/1e-4_lora_rk8.err
# MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/ArcC_Cloze LR=1e-4 BS=16 RANK=8 STEPS=20000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_0 2>>$ERR_0 

# # indenpendence
# OUT_1=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/ArcC_Cloze/llama_3.2_1b/2e-4_lora_rk8.out
# ERR_1=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/ArcC_Cloze/llama_3.2_1b/2e-4_lora_rk8.err
# MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/ArcC_Cloze LR=2e-4 BS=16 RANK=8 STEPS=20000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_1 2>>$ERR_1

# # indenpendence
# OUT_2=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/ArcC_Cloze/llama_3.2_1b/1e-4_lora_rk16.out
# ERR_2=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/ArcC_Cloze/llama_3.2_1b/1e-4_lora_rk16.err
# MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/ArcC_Cloze LR=1e-4 BS=16 RANK=16 STEPS=20000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_2 2>>$ERR_2

# # indenpendence
# OUT_3=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/ArcC_Cloze/llama_3.2_1b/2e-4_lora_rk16.out
# ERR_3=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/ArcC_Cloze/llama_3.2_1b/2e-4_lora_rk16.err
# MODEL=meta-llama/Llama-3.2-1B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/ArcC_Cloze LR=2e-4 BS=16 RANK=16 STEPS=20000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_3 2>>$ERR_3
