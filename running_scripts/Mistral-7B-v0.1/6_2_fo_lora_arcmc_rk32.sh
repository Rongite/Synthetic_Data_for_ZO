#!/bin/bash -l

# polaris 42h
cd /grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models


OUT_0=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/ArcC_MC/fo_lora/original/1e-4_lora_rk32.out
ERR_0=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/ArcC_MC/fo_lora/original/1e-4_lora_rk32.err
CUDA_VISIBLE_DEVICES=0,1 MODEL=mistralai/Mistral-7B-v0.1 MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/original/ArcC_MC LR=1e-4 BS=4 RANK=32 STEPS=80000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_0 2>>$ERR_0 &

OUT_1=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/ArcC_MC/fo_lora/original/2e-4_lora_rk32.out
ERR_1=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/ArcC_MC/fo_lora/original/2e-4_lora_rk32.err
CUDA_VISIBLE_DEVICES=2,3 MODEL=mistralai/Mistral-7B-v0.1 MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/original/ArcC_MC LR=2e-4 BS=4 RANK=32 STEPS=80000 SEED=0 bash fo_lora_finetune_original.sh 1>>$OUT_1 2>>$ERR_1 &

wait

OUT_2=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/ArcC_MC/fo_lora/rejection_sampling/1e-4_lora_rk32.out
ERR_2=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/ArcC_MC/fo_lora/rejection_sampling/1e-4_lora_rk32.err
CUDA_VISIBLE_DEVICES=0,1 MODEL=mistralai/Mistral-7B-v0.1 MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/ArcC_MC LR=1e-4 BS=4 RANK=32 STEPS=80000 SEED=0 bash fo_lora_finetune_synthetic.sh 1>>$OUT_2 2>>$ERR_2 &

OUT_3=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/ArcC_MC/fo_lora/rejection_sampling/2e-4_lora_rk32.out
ERR_3=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/results/Mistral-7B-v0.1/ArcC_MC/fo_lora/rejection_sampling/2e-4_lora_rk32.err
CUDA_VISIBLE_DEVICES=2,3 MODEL=mistralai/Mistral-7B-v0.1 MODE=ft TASK=/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/ArcC_MC LR=2e-4 BS=4 RANK=32 STEPS=80000 SEED=0 bash fo_lora_finetune_synthetic.sh 1>>$OUT_3 2>>$ERR_3 &

wait
