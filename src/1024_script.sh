#!/bin/bash
#SBATCH -J swinv2_res1024_cropped_allR  # Job name
#SBATCH -p gpu
#SBATCH -N 1 # Number of nodes
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-user=r.garridogarcia@northeastern.edu # Email
#SBATCH --mail-type=ALL # Type of email notifications

# Environment setup
module load anaconda3/2024.06 cuda/12.1.1 discovery

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate /projects/retprogression/pytorch_cu121_env

# Verify setup
echo "Python location: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Change to source directory
cd /home/r.garridogarcia/MIGHTE/retprogression/swinV2/RETProgression/src

# Force single GPU mode and set unique port
export CUDA_VISIBLE_DEVICES=0
export EXPERIMENT=experiment_res_1024
#export MASTER_PORT=$((12345 + RANDOM % 1000))  # Random port to avoid conflicts
# In your script, use:
RUN_ID=${1:-1}

# Auto-generate a free port
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

echo "Run ID: $RUN_ID"
echo "Using port: $MASTER_PORT"

# Run training
python train.py run_id=$RUN_ID exp.master_port=$MASTER_PORT