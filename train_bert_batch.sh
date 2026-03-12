#!/bin/bash
#SBATCH -A jdworak_jerry_ma_ad_ml_0001
#SBATCH -J BERT-TRM     # job name to display in squeue
#SBATCH -N 1            # number of nodes
#SBATCH -c 16           # request 16 cpus
#SBATCH -G 4            # request 4 gpus for multi-gpu training
#SBATCH --time=48:00:00 # 48 hours (adjust as needed)
#SBATCH --mem=256gb     # request 256GB node memory
#SBATCH --mail-user jerryma@smu.edu
#SBATCH --mail-type=end
#SBATCH --error=error_bert_trm_%j.log
#SBATCH --output=output_bert_trm_%j.log

set -e  # Exit on error

echo "======================================================================"
echo "BERT-TRM Training Pipeline"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "======================================================================"
echo

# Load required modules
module load conda gcc/11.2.0 cuda/11.8.0-vbvgppx

# Activate conda environment with torch2.8 and all dependencies
conda activate torch2.8
pip install -r requirements.txt

# Set working directory
cd /projects/jdworak/jerry_ma_AD_ML/idea_density/lms/bert-trm

# Set Python path
export PYTHONPATH=/projects/jdworak/jerry_ma_AD_ML/idea_density/lms/bert-trm:$PYTHONPATH

# Disable wandb if not needed (or configure it properly)
# To enable wandb, comment the next line and run: wandb login YOUR-LOGIN
export WANDB_MODE=disabled
export DISABLE_COMPILE=1

# Check if data exists, if not, prepare it
if [ ! -d "data/crawl-mlm" ]; then
    echo "======================================================================"
    echo "Step 1: Preparing Common Crawl data..."
    echo "======================================================================"
    
    # Download Common Crawl data if not exists
    if [ ! -f "data/common_crawl/crawl_text.txt" ]; then
        echo "Downloading Common Crawl text data..."
        python download_common_crawl.py
    fi
    
    # Build MLM dataset
    echo "Building MLM dataset from Common Crawl..."
    python -m dataset.build_text_mlm_dataset \
        --input-files data/common_crawl/crawl_text.txt \
        --output-dir data/crawl-mlm \
        --vocab-size 10000 \
        --max-seq-length 256 \
        --mask-prob 0.15 \
        --seed 42
    
    echo "Data preparation completed!"
    echo
else
    echo "Data already exists at data/crawl-mlm, skipping preparation"
    echo
fi

echo "======================================================================"
echo "Step 2: Training BERT model with multi-GPU setup..."
echo "======================================================================"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"
echo "Global batch size: 256"
echo "Configuration: BERT architecture with 4 layers, 256 hidden size"
echo "======================================================================"
echo

# Diagnostics

# Train with torchrun for multi-GPU distributed training
torchrun --nproc-per-node=$SLURM_GPUS_ON_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --nnodes=1 \
    pretrain.py \
    arch=bert \
    data_paths="[data/crawl-mlm]" \
    global_batch_size=256 \
    epochs=100000 \
    eval_interval=5000 \
    lr=1e-4 \
    arch.num_layers=4 \
    arch.hidden_size=256 \
    arch.num_heads=8 \
    arch.max_position_embeddings=256 \
    +run_name=bert_wiki_$(date +%Y%m%d_%H%M%S) \
    +project_name=bert-wiki-training \
    checkpoint_every_eval=True

echo
echo "======================================================================"
echo "Training completed!"
echo "End time: $(date)"
echo "======================================================================"
echo
echo "Checkpoints saved in: outputs/bert/"
echo "Logs saved in: output_bert_trm_${SLURM_JOB_ID}.log"
echo
