NUM_MACHINES=1
NUM_LOCAL_GPUS=8
MACHINE_RANK=0

export WANDB_API_KEY="" # Modify this if you use wandb

accelerate launch \
    --num_machines $NUM_MACHINES \
    --num_processes $(( $NUM_MACHINES * $NUM_LOCAL_GPUS )) \
    --machine_rank $MACHINE_RANK \
    src/train_partcrafter.py \
        --pin_memory \
        --allow_tf32 \
$@
