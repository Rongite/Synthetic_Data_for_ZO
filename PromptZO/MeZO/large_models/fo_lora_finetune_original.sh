#!/usr/bin/env bash

# ========================
# 0) 说明
# - 通过环境变量 RANK 来控制 LoRA 的秩 (默认 16)。
# - 通过环境变量 LR   来控制微调时的学习率 (默认 1e-5)。
# - 其他如 STEPS、BS、SEED、TRAIN、DEV、EVAL 等，也可通过环境变量覆盖。
# ========================

# [CHANGED] 这里保留对 RANK / LR 的环境变量引用，通过 export 或命令行赋值即可
RANK=${RANK:-16}    # <-- LoRA rank，可在执行脚本前 export RANK=32 等
LR=${LR:-1e-5}      # <-- 学习率，可在执行脚本前 export LR=5e-5 等

# 其余变量如 MODEL、STEPS、BS 等也可覆盖
MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

STEPS=${STEPS:-20000}
BS=${BS:-16}
EPS=${EPS:-1e-3}
SEED=${SEED:-0}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}

# [CHANGED] 我们这里固定 MODE=lora，用于 LoRA 微调
MODE=lora           # <-- 如果需要手动切换，也可在执行脚本前 export MODE=lora
EXTRA_ARGS=""

# [CHANGED] TAG 里附加 lora rank / lr 信息，以便区分不同实验
TAG=firstorder-$MODEL-$TASK-$MODE-$BS-$LR-$SEED-lora-$RANK

if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
    TAG=firstorder-$MODEL-$TASK-$MODE-$BS-$LR-$SEED-prefix-5
elif [ "$MODE" == "lora" ]; then
    # [CHANGED] 仅在 lora 模式下启用 --lora --lora_rank
    EXTRA_ARGS="--lora --lora_rank $RANK"
    TAG=firstorder-$MODEL-$TASK-$MODE-$BS-$LR-$SEED-lora-$RANK
fi

# 如果 TASK 是以路径形式指定，提取其中最后的目录名作为 TASK_NAME
if [[ "$TASK" == */* ]]; then
    TASK_NAME=$(basename "$TASK")
else
    TASK_NAME="$TASK"
fi

TASK_ARGS=""
case $TASK_NAME in
    CB)
        DEV=100
        ;;
    Copa)
        DEV=100
        TASK_ARGS="--train_as_classification False"
        ;;
    DROP)
        TASK_ARGS="--train_as_classification False"
        ;;
    SQuAD)
        TASK_ARGS="--train_as_classification False"
        ;;
esac

echo $TAG
echo "STEPS: $STEPS"
echo "BS: $BS"
echo "LR: $LR"            # [CHANGED] 打印出当前学习率
echo "EPS: $EPS"
echo "SEED: $SEED"
echo "MODE: $MODE"
echo "LoRA RANK: $RANK"  # [CHANGED] 打印出当前 LoRA rank
echo "Extra args: $EXTRA_ARGS $TASK_ARGS"

# # siai
# python run.py \
#     --model_name $MODEL \
#     --task_name $TASK \
#     --output_dir /home/jlong1/Downloads/models/synthetic_data/fo_lora/original/$TASK_NAME-${MODEL_NAME}-$TAG \
#     --tag $TAG \
#     --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL \
#     --logging_steps 10 \
#     --max_steps $STEPS \
#     --trainer regular \
#     --fp16 \
#     --learning_rate $LR \
#     --per_device_train_batch_size $BS \
#     --load_best_model_at_end \
#     --evaluation_strategy steps \
#     --save_strategy steps \
#     --save_total_limit 1 \
#     --train_as_classification \
#     --eval_steps 500 \
#     --save_steps 500 \
#     --record_time \
#     $EXTRA_ARGS \
#     $TASK_ARGS \
#     "$@"

# # gh200
# python run.py \
#     --model_name $MODEL \
#     --task_name $TASK \
#     --output_dir /home/ubuntu/LLM-inference/jikai-project/models/synthetic_data/fo_lora/original/$TASK_NAME-${MODEL_NAME}-$TAG \
#     --tag $TAG \
#     --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL \
#     --logging_steps 10 \
#     --max_steps $STEPS \
#     --trainer regular \
#     --fp16 \
#     --learning_rate $LR \
#     --per_device_train_batch_size $BS \
#     --load_best_model_at_end \
#     --evaluation_strategy steps \
#     --save_strategy steps \
#     --save_total_limit 1 \
#     --train_as_classification \
#     --eval_steps 500 \
#     --save_steps 500 \
#     --record_time \
#     $EXTRA_ARGS \
#     $TASK_ARGS \
#     "$@"

# polaris
python run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir /grand/sbi-fair/jikaiLoong/models/synthetic_data/fo_lora/original/$TASK_NAME-${MODEL_NAME}-$TAG \
    --tag $TAG \
    --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL \
    --logging_steps 10 \
    --max_steps $STEPS \
    --trainer regular \
    --fp16 \
    --learning_rate $LR \
    --per_device_train_batch_size $BS \
    --load_best_model_at_end \
    --evaluation_strategy steps \
    --save_strategy steps \
    --save_total_limit 1 \
    --train_as_classification \
    --eval_steps 500 \
    --save_steps 500 \
    --record_time \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@"
