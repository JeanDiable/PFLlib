#!/bin/bash

# APOP Experiment Runner with Generated Task Sequences
# Usage: ./run_apop.sh [sequence_file] [dataset] [num_clients] [cilrpc]

# Configuration
DATASET=${2:-"Cifar100"}  # Fixed: Use Cifar10 to match actual data
NUM_CLIENTS=${3:-10}
CILRPC=${4:-20}  # Rounds per task (main parameter to adjust)
SEQUENCE_FILE=${1:-""}

# Task sequence configuration (adjusted for Cifar10)
NUM_TASKS_PER_CLIENT=10  # Reduced to fit in 10 classes
CLASSES_PER_TASK=10      # Use 2 classes per task to fit better

# Dataset-specific settings
case $DATASET in
    "Cifar10")
        TOTAL_CLASSES=10
        ;;
    "Cifar100")
        TOTAL_CLASSES=100
        ;;
    "MNIST")
        TOTAL_CLASSES=10
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        exit 1
        ;;
esac

# Calculate total rounds: num_tasks_per_client × rounds_per_task
TOTAL_ROUNDS=$((NUM_TASKS_PER_CLIENT * CILRPC))

echo "=== APOP Experiment Configuration ==="
echo "Dataset: $DATASET (${TOTAL_CLASSES} classes)"
echo "Clients: $NUM_CLIENTS"
echo "Tasks per client: $NUM_TASKS_PER_CLIENT"
echo "Classes per task: $CLASSES_PER_TASK"
echo "Rounds per task (cilrpc): $CILRPC"
echo "Total rounds: $TOTAL_ROUNDS"
echo "Sequence file: $SEQUENCE_FILE"

# Generate or use provided task sequences
if [[ -z "$SEQUENCE_FILE" || ! -f "$SEQUENCE_FILE" ]]; then
    echo "Generating task sequences..."
    SEQUENCE_FILE="generated_sequences_${DATASET}_${NUM_CLIENTS}c_${NUM_TASKS_PER_CLIENT}t_${CLASSES_PER_TASK}cpt.txt"
    python generate_task_sequence.py \
        -nc $NUM_CLIENTS \
        -ntpc $NUM_TASKS_PER_CLIENT \
        -cpt $CLASSES_PER_TASK \
        -tc $TOTAL_CLASSES \
        -o $SEQUENCE_FILE
    
    if [[ $? -ne 0 ]]; then
        echo "Failed to generate sequences"
        exit 1
    fi
    echo "Generated: $SEQUENCE_FILE"
else
    echo "Using existing sequence file: $SEQUENCE_FILE"
fi

# Read sequences from file
CLIENT_SEQUENCES=$(cat $SEQUENCE_FILE)

echo "Running APOP experiment..."
echo "Client sequences: $CLIENT_SEQUENCES"

cd system

python main.py \
    -data $DATASET \
    -m ResNet18 \
    -algo APOP \
    -gr $TOTAL_ROUNDS \
    -nc $NUM_CLIENTS \
    -ncl $TOTAL_CLASSES \
    -cil True \
    -til True \
    -pfcl True \
    -client_seq "$CLIENT_SEQUENCES" \
    -cilrpc $CILRPC \
    -subspace_dim 25 \
    -adaptation_threshold 0.3 \
    -fusion_threshold 0.4 \
    -max_transfer_gain 0 \
    -min_adaptation_rounds 5 \
    -lr 0.01 \
    -lbs 32 \
    -ls 5\
    -eg 3 \
    -dev cuda \
    -did 0 \
    -wandb True \
    -wandb_project "iclr26" \
    -go "${DATASET}_${NUM_CLIENTS}c_${NUM_TASKS_PER_CLIENT}t_${CILRPC}rpt_orth"

echo "Experiment completed!"
