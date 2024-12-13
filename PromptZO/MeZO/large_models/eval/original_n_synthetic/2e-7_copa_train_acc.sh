python /grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models/eval_acc.py \
    --model_name /grand/sbi-fair/jikaiLoong/models/synthetic_data/zo/original_n_synthetic/Copa-Llama-3.2-1B-mezo-meta-llama/Llama-3.2-1B-/grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/original_n_synthetic/mezo/Copa-ft-16-2e-7-1e-3-0/checkpoint-18000 \
    --task_name /grand/sbi-fair/jikaiLoong/Synthetic_Data_for_ZO/Data/original_n_synthetic/mezo/Copa \
    --output_dir ./result \
    --num_train 1000 \
    --num_dev 100 \
    --trainer none \
    --num_eval 1000 \
    --train_set_seed 0