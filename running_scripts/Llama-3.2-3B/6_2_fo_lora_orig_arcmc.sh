#!/bin/bash -l

# 
cd /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models

OUT_0=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-3B/ArcC_MC/fo_lora/original/1e-4_lora_rk8.out
ERR_0=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-3B/ArcC_MC/fo_lora/original/1e-4_lora_rk8.err
MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/ArcC_MC LR=1e-4 BS=4 RANK=8 STEPS=20000 SEED=0 bash fo_lora_finetune_original.sh --gradient_accumulation_steps 4 1>>$OUT_0 2>>$ERR_0 &

OUT_1=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-3B/ArcC_MC/fo_lora/original/2e-4_lora_rk8.out
ERR_1=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-3B/ArcC_MC/fo_lora/original/2e-4_lora_rk8.err
MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/ArcC_MC LR=2e-4 BS=4 RANK=8 STEPS=20000 SEED=0 bash fo_lora_finetune_original.sh --gradient_accumulation_steps 4 1>>$OUT_1 2>>$ERR_1 &

OUT_2=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-3B/ArcC_MC/fo_lora/original/1e-4_lora_rk16.out
ERR_2=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-3B/ArcC_MC/fo_lora/original/1e-4_lora_rk16.err
MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/ArcC_MC LR=1e-4 BS=4 RANK=16 STEPS=20000 SEED=0 bash fo_lora_finetune_original.sh --gradient_accumulation_steps 4 1>>$OUT_2 2>>$ERR_2 &

OUT_3=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-3B/ArcC_MC/fo_lora/original/2e-4_lora_rk16.out
ERR_3=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-3B/ArcC_MC/fo_lora/original/2e-4_lora_rk16.err
MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/ArcC_MC LR=2e-4 BS=4 RANK=16 STEPS=20000 SEED=0 bash fo_lora_finetune_original.sh --gradient_accumulation_steps 4 1>>$OUT_3 2>>$ERR_3 &

wait

OUT_4=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-3B/ArcC_MC/fo_lora/original/1e-4_lora_rk32.out
ERR_4=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-3B/ArcC_MC/fo_lora/original/1e-4_lora_rk32.err
MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/ArcC_MC LR=1e-4 BS=4 RANK=32 STEPS=20000 SEED=0 bash fo_lora_finetune_original.sh --gradient_accumulation_steps 4 1>>$OUT_4 2>>$ERR_4 &

OUT_5=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-3B/ArcC_MC/fo_lora/original/2e-4_lora_rk32.out
ERR_5=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/Llama-3.2-3B/ArcC_MC/fo_lora/original/2e-4_lora_rk32.err
MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/ArcC_MC LR=2e-4 BS=4 RANK=32 STEPS=20000 SEED=0 bash fo_lora_finetune_original.sh --gradient_accumulation_steps 4 1>>$OUT_5 2>>$ERR_5 &

# wait


# siai
# cd /home/jlong1/Downloads/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models

# # indenpendence
# OUT_0=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/ArcC_MC/llama_3.2_1b/1e-4_lora_rk8.out
# ERR_0=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/ArcC_MC/llama_3.2_1b/1e-4_lora_rk8.err
# MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/ArcC_MC LR=1e-4 BS=4 RANK=8 STEPS=20000 SEED=0 bash fo_lora_finetune_original.sh --gradient_accumulation_steps 4 1>>$OUT_0 2>>$ERR_0 

# # indenpendence
# OUT_1=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/ArcC_MC/llama_3.2_1b/2e-4_lora_rk8.out
# ERR_1=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/ArcC_MC/llama_3.2_1b/2e-4_lora_rk8.err
# MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/ArcC_MC LR=2e-4 BS=4 RANK=8 STEPS=20000 SEED=0 bash fo_lora_finetune_original.sh --gradient_accumulation_steps 4 1>>$OUT_1 2>>$ERR_1

# # indenpendence
# OUT_2=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/ArcC_MC/llama_3.2_1b/1e-4_lora_rk16.out
# ERR_2=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/ArcC_MC/llama_3.2_1b/1e-4_lora_rk16.err
# MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/ArcC_MC LR=1e-4 BS=4 RANK=16 STEPS=20000 SEED=0 bash fo_lora_finetune_original.sh --gradient_accumulation_steps 4 1>>$OUT_2 2>>$ERR_2

# # indenpendence
# OUT_3=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/ArcC_MC/llama_3.2_1b/2e-4_lora_rk16.out
# ERR_3=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/3_first_order_lora/result/ArcC_MC/llama_3.2_1b/2e-4_lora_rk16.err
# MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/ArcC_MC LR=2e-4 BS=4 RANK=16 STEPS=20000 SEED=0 bash fo_lora_finetune_original.sh --gradient_accumulation_steps 4 1>>$OUT_3 2>>$ERR_3
