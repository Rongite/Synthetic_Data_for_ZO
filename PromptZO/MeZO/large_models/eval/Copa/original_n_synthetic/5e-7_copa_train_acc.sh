python /home/jlong1/Downloads/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models/eval_acc.py \
    --model_name /home/jlong1/Downloads/models/synthetic_data/zo/original_n_synthetic/Copa-Llama-3.2-1B-mezo-meta-llama/Llama-3.2-1B-/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original_n_synthetic/mezo/Copa-ft-16-5e-7-1e-3-0/checkpoint-8500 \
    --task_name /home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original_n_synthetic/mezo/Copa \
    --output_dir ./result \
    --num_train 1000 \
    --num_dev 100 \
    --trainer none \
    --num_eval 1000 \
    --train_set_seed 0