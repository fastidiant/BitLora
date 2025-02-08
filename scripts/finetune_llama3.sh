MODEL_PATH=meta-llama/Meta-Llama-3-8B
OUTPUT_DIR=./lora_checkpoints
TRAIN_DATA_PATH=../data/
BATCH_SIZE=
VAL_SPLIT=
GRAD_ACC_STEPS=
MAX_LENGTH=1024
NUM_EPOCHS=3
WARMUP_STEPS=100
SAVE_STEPS=
EVAL_STEP=
EVAL_DELAY=
LORA_TYPE=aplinear
LORA_RANK=32
LORA_ALPHA=64
WANDB_PROJECT=
WANDB_ENTITY=
WANDB_RUN_NAME=

LEARNING_RATES="4e-4"
SCALE_LR=1e-5

echo "Running training with --lora aplinear and learning_rate $LEARNING_RATE..."

python ../train.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIRS" \
    --train_data_path "$TRAIN_DATA_PATH" \
    --batch_size "$BATCH_SIZE" \
    --val_split "$VAL_SPLIT" \
    --grad_acc_steps "$GRAD_ACC_STEPS" \
    --max_length "$MAX_LENGTH" \
    --num_epochs "$NUM_EPOCHS" \
    --warmup_steps "$WARMUP_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --eval_steps "$EVAL_STEP" \
    --eval_delay "$EVAL_DELAY" \
    --learning_rate "$LEARNING_RATE" \
    --scale_lr "$SCALE_LR" \
    --lora_type "$LORA_TYPE" \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_run_name "$WANDB_RUN_NAME"
