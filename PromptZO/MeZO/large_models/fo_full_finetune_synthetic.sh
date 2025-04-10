MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

STEPS=${STEPS:-20000}
RANK=${RANK:-16}
BS=${BS:-16}
LR=${LR:-1e-5}
EPS=${EPS:-1e-3}
SEED=${SEED:-0}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}

MODE=${MODE:-ft}
EXTRA_ARGS=""

TAG=firstorder-$MODEL-$TASK-$MODE-$BS-$LR-$SEED

if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
    TAG=firstorder-$MODEL-$TASK-$MODE-$BS-$LR-$SEED-prefix-5
elif [ "$MODE" == "lora" ]; then
    EXTRA_ARGS="--lora --lora_rank $RANK"
    TAG=firstorder-$MODEL-$TASK-$MODE-$BS-$LR-$SEED-lora-$RANK
fi

# Extract the task name
if [[ "$TASK" == */* ]]; then
    TASK_NAME=$(basename $TASK)
else
    TASK_NAME=$TASK
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
echo "LR: $LR"
echo "EPS: $EPS"
echo "SEED: $SEED"
echo "MODE: $MODE"
echo "Extra args: $EXTRA_ARGS $TASK_ARGS"

python run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir /home/jlong1/Downloads/models/synthetic_data/fo_full/synthetic/$TASK_NAME-${MODEL_NAME}-$TAG \
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
