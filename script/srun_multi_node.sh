#!/bin/bash

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <num_nodes> <num_gpus_per_node> [your command after torchrun]..."
    exit 1
fi

JOB_NAME=$USER-chitu
NODES=$1
NTASKS_PER_NODE=1
CPUS_PER_TASK=32
GPUS_PER_TASK=$2

THIS_SCRIPT=$(realpath $0)

if [[ "$3" != "--node" ]]; then
    COMMAND=${@:3}
    PARAMS="--job-name $JOB_NAME --nodes $NODES --ntasks-per-node $NTASKS_PER_NODE --cpus-per-task $CPUS_PER_TASK --gpus-per-task $GPUS_PER_TASK"
    exec srun $PARAMS $THIS_SCRIPT $1 $2 --node $COMMAND
fi

COMMAND=${@:4}

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$(($SLURM_JOB_ID+52000))
RDVZ_PORT=$(($SLURM_JOB_ID+53000))
RDVZ_ID=chitu

echo prepare torchrun on node $(hostname) 

torchrun \
    --nnodes $SLURM_NNODES \
    --nproc-per-node $SLURM_GPUS_ON_NODE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --rdzv-endpoint $MASTER_ADDR:$RDVZ_PORT \
    --rdzv-backend=c10d \
    --rdzv-id $RDVZ_ID \
    $COMMAND
