#!/bin/bash
#SBATCH -J swinv2_res384 # Job name
#SBATCH -p gpu
#SBATCH -N 1 # Number of nodes
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-cpu=16G
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
export EXPERIMENT=experiment_res_384
export MASTER_PORT=$((12345 + RANDOM % 1000))  # Random port to avoid conflicts

echo "Using port: $MASTER_PORT"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"

# Run training
python train.py