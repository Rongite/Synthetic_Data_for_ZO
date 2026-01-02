#!/usr/bin/env bash

# ========================
# 0) Description
# - Use environment variable RANK to control LoRA rank (default 16).
# - Use environment variable LR to control learning rate for fine-tuning (default 1e-5).
# - Other variables like STEPS, BS, SEED, TRAIN, DEV, EVAL can also be overridden via environment variables.
# ========================

# [CHANGED] Preserve environment variable references for RANK / LR, can be set via export or command line
RANK=${RANK:-16}    # <-- LoRA rank, can export RANK=32 before running script, etc.
LR=${LR:-1e-5}      # <-- Learning rate, can export LR=5e-5 before running script, etc.

# Other variables like MODEL, STEPS, BS can also be overridden
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

# [CHANGED] Fix MODE=lora here for LoRA fine-tuning
MODE=lora           # <-- Can also export MODE=lora before running script if manual switching is needed
EXTRA_ARGS=""

# [CHANGED] Append lora rank / lr information to TAG to distinguish different experiments
TAG=firstorder-$MODEL-$TASK-$MODE-$BS-$LR-$SEED-lora-$RANK

if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
    TAG=firstorder-$MODEL-$TASK-$MODE-$BS-$LR-$SEED-prefix-5
elif [ "$MODE" == "lora" ]; then
    # [CHANGED] Enable --lora --lora_rank only in lora mode
    EXTRA_ARGS="--lora --lora_rank $RANK"
    TAG=firstorder-$MODEL-$TASK-$MODE-$BS-$LR-$SEED-lora-$RANK
fi

# If TASK is specified as a path, extract the last directory name as TASK_NAME
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
echo "LR: $LR"            # [CHANGED] Print current learning rate
echo "EPS: $EPS"
echo "SEED: $SEED"
echo "MODE: $MODE"
echo "LoRA RANK: $RANK"  # [CHANGED] Print current LoRA rank
echo "Extra args: $EXTRA_ARGS $TASK_ARGS"

# # siai
# python run.py \
#     --model_name $MODEL \
#     --task_name $TASK \
#     --output_dir /home/jlong1/Downloads/models/synthetic_data/fo_lora/synthetic/$TASK_NAME-${MODEL_NAME}-$TAG \
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

# gh200
python run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir /home/ubuntu/LLM-inference/jikai-project/models/synthetic_data/fo_lora/synthetic/$TASK_NAME-${MODEL_NAME}-$TAG \
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

# # polaris
# python run.py \
#     --model_name $MODEL \
#     --task_name $TASK \
#     --output_dir /grand/sbi-fair/jikaiLoong/models/synthetic_data/fo_lora/synthetic/$TASK_NAME-${MODEL_NAME}-$TAG \
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
