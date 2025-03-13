#!/bin/bash

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <docker-container-name> <pwd-in-container> <comma-separated-hosts> <num_gpus_per_node> [your command after torchrun]..."
    exit 1
fi

DOCKER_CONTAINER_NAME=$1
PWD_IN_CONTAINER=$2
NODE_COMMA_SEP_LIST=$3
GPUS_PER_TASK=$4

NUM_NODES=$(echo $NODE_COMMA_SEP_LIST | tr "," "\n" | wc -l)
MASTER_NODE=$(echo $NODE_COMMA_SEP_LIST | cut -d',' -f1)
MASTER_PORT=52000
RDVZ_PORT=53000
RDVZ_ID=chitu

COMMAND=${@:5}
TORCHRUN_CMD="torchrun \
    --nnodes $NUM_NODES \
    --nproc_per_node $GPUS_PER_TASK \
    --master_addr $MASTER_NODE \
    --master_port $MASTER_PORT \
    --rdzv_endpoint $MASTER_NODE:$RDVZ_PORT \
    --rdzv_backend=c10d \
    --rdzv_id $RDVZ_ID \
    $COMMAND"
DOCKER_CMD="docker exec -t $DOCKER_CONTAINER_NAME bash --login -c \"cd $PWD_IN_CONTAINER; $TORCHRUN_CMD\""
echo "Command on each node: $DOCKER_CMD"

# Kill all background jobs on SIGINT
trap 'kill $(jobs -p)' SIGINT

IFS=',' # Comma is set as delimiter
for node in $NODE_COMMA_SEP_LIST; do
    # Use `&` instead of `-f` to run the command in the background, so we can wait form them
    ssh -n $node $DOCKER_CMD &
done
unset IFS

wait # Wait for all ssh commands to finish
