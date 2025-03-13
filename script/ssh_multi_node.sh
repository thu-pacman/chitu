#!/bin/bash

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <comma-separated-hosts> <num_gpus_per_node> [your command after torchrun]..."
    exit 1
fi

NODE_COMMA_SEP_LIST=$1
GPUS_PER_TASK=$2

NUM_NODES=$(echo $NODE_COMMA_SEP_LIST | tr "," "\n" | wc -l)
MASTER_NODE=$(echo $NODE_COMMA_SEP_LIST | cut -d',' -f1)
MASTER_PORT=52000
RDVZ_PORT=53000
RDVZ_ID=chitu

COMMAND=${@:3}
TORCHRUN_CMD="torchrun \
    --nnodes $NUM_NODES \
    --nproc_per_node $GPUS_PER_TASK \
    --master_addr $MASTER_NODE \
    --master_port $MASTER_PORT \
    --rdzv_endpoint $MASTER_NODE:$RDVZ_PORT \
    --rdzv_backend=c10d \
    --rdzv_id $RDVZ_ID \
    $COMMAND"

# Kill all background jobs on SIGINT
trap 'kill $(jobs -p)' SIGINT

IFS=',' # Comma is set as delimiter
for node in $NODE_COMMA_SEP_LIST; do
    # Use `&` instead of `-f` to run the command in the background, so we can wait form them
    ssh -n $node "\
        PATH=$PATH \
        LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
        VIRTUAL_ENV=$VIRTUAL_ENV \
        NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME \
        NCCL_P2P_LEVEL=$NCCL_P2P_LEVEL \
        NCCL_IB_DISABLE=$NCCL_IB_DISABLE \
        NCCL_IB_TIMEOUT=$NCCL_IB_TIMEOUT \
        NCCL_IB_RETRY_CNT=$NCCL_IB_RETRY_CNT \
        NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX \
        NCCL_IB_HCA=$NCCL_IB_HCA \
        TP_SOCKET_IFNAME=$TP_SOCKET_IFNAME \
        GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME \
        NCCL_DEBUG=$NCCL_DEBUG \
        bash -c \"cd $(pwd); $TORCHRUN_CMD\"" &
done
unset IFS

wait # Wait for all ssh commands to finish
