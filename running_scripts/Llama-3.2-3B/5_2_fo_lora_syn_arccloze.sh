#!/bin/bash -l

# polaris 16h
cd /grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models

OUT_0=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Llama-3.2-3B/ArcC_Cloze/fo_lora/rejection_sampling/1e-4_lora_rk8.out
ERR_0=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Llama-3.2-3B/ArcC_Cloze/fo_lora/rejection_sampling/1e-4_lora_rk8.err
CUDA_VISIBLE_DEVICES=0 MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/ArcC_Cloze LR=1e-4 BS=4 RANK=8 STEPS=20000 SEED=0 bash fo_lora_finetune_synthetic.sh 1>>$OUT_0 2>>$ERR_0 &

OUT_1=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Llama-3.2-3B/ArcC_Cloze/fo_lora/rejection_sampling/2e-4_lora_rk8.out
ERR_1=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Llama-3.2-3B/ArcC_Cloze/fo_lora/rejection_sampling/2e-4_lora_rk8.err
CUDA_VISIBLE_DEVICES=1 MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/ArcC_Cloze LR=2e-4 BS=4 RANK=8 STEPS=20000 SEED=0 bash fo_lora_finetune_synthetic.sh 1>>$OUT_1 2>>$ERR_1 &

OUT_2=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Llama-3.2-3B/ArcC_Cloze/fo_lora/rejection_sampling/1e-4_lora_rk16.out
ERR_2=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Llama-3.2-3B/ArcC_Cloze/fo_lora/rejection_sampling/1e-4_lora_rk16.err
CUDA_VISIBLE_DEVICES=2 MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/ArcC_Cloze LR=1e-4 BS=4 RANK=16 STEPS=20000 SEED=0 bash fo_lora_finetune_synthetic.sh 1>>$OUT_2 2>>$ERR_2 &

OUT_3=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Llama-3.2-3B/ArcC_Cloze/fo_lora/rejection_sampling/2e-4_lora_rk16.out
ERR_3=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Llama-3.2-3B/ArcC_Cloze/fo_lora/rejection_sampling/2e-4_lora_rk16.err
CUDA_VISIBLE_DEVICES=3 MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/ArcC_Cloze LR=2e-4 BS=4 RANK=16 STEPS=20000 SEED=0 bash fo_lora_finetune_synthetic.sh 1>>$OUT_3 2>>$ERR_3 &

wait

OUT_4=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Llama-3.2-3B/ArcC_Cloze/fo_lora/rejection_sampling/1e-4_lora_rk32.out
ERR_4=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Llama-3.2-3B/ArcC_Cloze/fo_lora/rejection_sampling/1e-4_lora_rk32.err
CUDA_VISIBLE_DEVICES=0 MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/ArcC_Cloze LR=1e-4 BS=4 RANK=32 STEPS=20000 SEED=0 bash fo_lora_finetune_synthetic.sh 1>>$OUT_4 2>>$ERR_4 &

OUT_5=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Llama-3.2-3B/ArcC_Cloze/fo_lora/rejection_sampling/2e-4_lora_rk32.out
ERR_5=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Llama-3.2-3B/ArcC_Cloze/fo_lora/rejection_sampling/2e-4_lora_rk32.err
CUDA_VISIBLE_DEVICES=2 MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/ArcC_Cloze LR=2e-4 BS=4 RANK=32 STEPS=20000 SEED=0 bash fo_lora_finetune_synthetic.sh 1>>$OUT_5 2>>$ERR_5 &

wait

# # siai
# cd /home/jlong1/Downloads/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models

# OUT_0=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/rejection_sampling/3_first_order_lora/result/ArcC_Cloze/llama_3.2_1b/1e-4_lora_rk8.out
# ERR_0=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/rejection_sampling/3_first_order_lora/result/ArcC_Cloze/llama_3.2_1b/1e-4_lora_rk8.err
# CUDA_VISIBLE_DEVICES=0 MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/ArcC_Cloze LR=1e-4 BS=4 RANK=8 STEPS=20000 SEED=0 bash fo_lora_finetune_synthetic.sh 1>>$OUT_0 2>>$ERR_0 &

# OUT_2=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/rejection_sampling/3_first_order_lora/result/ArcC_Cloze/llama_3.2_1b/1e-4_lora_rk16.out
# ERR_2=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/rejection_sampling/3_first_order_lora/result/ArcC_Cloze/llama_3.2_1b/1e-4_lora_rk16.err
# CUDA_VISIBLE_DEVICES=1 MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/ArcC_Cloze LR=1e-4 BS=4 RANK=16 STEPS=20000 SEED=0 bash fo_lora_finetune_synthetic.sh 1>>$OUT_2 2>>$ERR_2 &

# wait

# OUT_1=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/rejection_sampling/3_first_order_lora/result/ArcC_Cloze/llama_3.2_1b/2e-4_lora_rk8.out
# ERR_1=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/rejection_sampling/3_first_order_lora/result/ArcC_Cloze/llama_3.2_1b/2e-4_lora_rk8.err
# CUDA_VISIBLE_DEVICES=0 MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/ArcC_Cloze LR=2e-4 BS=4 RANK=8 STEPS=20000 SEED=0 bash fo_lora_finetune_synthetic.sh 1>>$OUT_1 2>>$ERR_1 &

# OUT_3=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/rejection_sampling/3_first_order_lora/result/ArcC_Cloze/llama_3.2_1b/2e-4_lora_rk16.out
# ERR_3=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/rejection_sampling/3_first_order_lora/result/ArcC_Cloze/llama_3.2_1b/2e-4_lora_rk16.err
# CUDA_VISIBLE_DEVICES=1 MODEL=meta-llama/Llama-3.2-3B MODE=ft TASK=/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/ArcC_Cloze LR=2e-4 BS=4 RANK=16 STEPS=20000 SEED=0 bash fo_lora_finetune_synthetic.sh 1>>$OUT_3 2>>$ERR_3 &

# wait

