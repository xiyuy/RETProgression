#!/bin/bash
#SBATCH -J val_inference_complete_test  # Job name
#SBATCH -p gpu                           # GPU partition
#SBATCH -N 1                             # Number of nodes
#SBATCH --time=6:00:00                   # 4 hours should be enough for GPU
#SBATCH --gres=gpu:1                     # Request 1 GPU (any available)
#SBATCH --cpus-per-task=8                # CPUs for data loading
#SBATCH --mem-per-cpu=16G                # Memory per CPU
#SBATCH --mail-user=r.garridogarcia@northeastern.edu
#SBATCH --mail-type=ALL                  # Email on start, end, fail
#SBATCH --output=validation_%j.out       # Standard output log
#SBATCH --error=validation_%j.err        # Standard error log

# Print job information
echo "========================================"
echo "SLURM Job Information"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================"

# Environment setup
module load anaconda3/2024.06 cuda/12.1.1 discovery

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate /projects/retprogression/pytorch_cu121_env

# Verify GPU setup
echo "Environment Check:"
echo "Python location: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "GPU name: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0
echo "Using CUDA device: $CUDA_VISIBLE_DEVICES"

# Change to the directory with your scripts
cd /home/r.garridogarcia/MIGHTE/retprogression/swinV2/RETProgression/results_analysis

# Configuration
CHECKPOINT_PATH="/projects/retprogression/rgarridogarcia/checkpoints/clean_gradable_swinv2_resolution1024_oversample50_RetinaCropped_BrightnessFilter/best_balanced_acc_model.pth"
DATA_DIR="/projects/retprogression"
ANNOTATIONS_FILE="clean_gradable_dr_val.csv"
IMG_DIR="cropped1024_brightness_noresize_dataset_08202025" #"complete_cropped1024_brightness_09112025"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/projects/retprogression/rgarridogarcia/clean_gradable_swinv2_resolution1024_oversample50_RetinaCropped_BrightnessFilter/val_results_${TIMESTAMP}"

# Model configuration
MODEL_NAME="swinv2_large_window12to16_192to256.ms_in22k_ft_in1k"
IMG_SIZE=1024
BATCH_SIZE=16  # Reasonable for GPU with 1024x1024 images
NUM_WORKERS=4  # For efficient data loading

# Print configuration
echo ""
echo "========================================"
echo "VALIDATION CONFIGURATION"
echo "========================================"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Data directory: $DATA_DIR"
echo "Annotations file: $ANNOTATIONS_FILE"
echo "Image directory: $IMG_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Model: $MODEL_NAME"
echo "Image size: $IMG_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Workers: $NUM_WORKERS"
echo "========================================"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

# Check if annotations file exists
if [ ! -f "$DATA_DIR/$ANNOTATIONS_FILE" ]; then
    echo "ERROR: Annotations file not found at $DATA_DIR/$ANNOTATIONS_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Save configuration to output directory
cat > "${OUTPUT_DIR}/config.txt" << EOF
Validation Configuration
========================
Date: $(date)
Job ID: $SLURM_JOB_ID
Node: $SLURM_NODELIST
Checkpoint: $CHECKPOINT_PATH
Data Directory: $DATA_DIR
Annotations: $ANNOTATIONS_FILE
Image Directory: $IMG_DIR
Model: $MODEL_NAME
Image Size: $IMG_SIZE
Batch Size: $BATCH_SIZE
Workers: $NUM_WORKERS
Output Directory: $OUTPUT_DIR
EOF

echo ""
echo "Starting validation inference..."
echo ""

# Change to the directory with your scripts
cd /home/r.garridogarcia/MIGHTE/retprogression/swinV2/RETProgression/results_analysis

# Add src directory to Python path so imports work
export PYTHONPATH="/home/r.garridogarcia/MIGHTE/retprogression/swinV2/RETProgression/src:${PYTHONPATH}"
echo "Python path: $PYTHONPATH"

# Now run the script (it should find custom_metrics now)
python validation_inference.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --model_name "$MODEL_NAME" \
    --data_dir "$DATA_DIR" \
    --annotations_file "$ANNOTATIONS_FILE" \
    --img_dir "$IMG_DIR" \
    --id_column "ID" \
    --label_column "gradable_binary_DR" \
    --img_size $IMG_SIZE \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --output_dir "$OUTPUT_DIR" \
    --device cuda \
    --no_resize
# Check exit status
EXIT_STATUS=$?

if [ $EXIT_STATUS -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "VALIDATION COMPLETED SUCCESSFULLY!"
    echo "========================================"
    echo "End Time: $(date)"
    echo "Output directory: $OUTPUT_DIR"
    echo ""
    echo "Output files:"
    ls -lh "$OUTPUT_DIR"/*.csv 2>/dev/null || echo "No CSV files generated"
    
    # Show preview of results if available
    PRED_FILE="${OUTPUT_DIR}/predictions_*.csv"
    if ls $PRED_FILE 1> /dev/null 2>&1; then
        echo ""
        echo "Predictions summary:"
        wc -l $PRED_FILE
    fi
    
    METRICS_FILE="${OUTPUT_DIR}/metrics_*.csv"
    if ls $METRICS_FILE 1> /dev/null 2>&1; then
        echo ""
        echo "Performance metrics:"
        cat $METRICS_FILE
    fi
else
    echo ""
    echo "========================================"
    echo "ERROR: Validation failed with exit code $EXIT_STATUS"
    echo "========================================"
    echo "Check the error logs above for details"
    exit $EXIT_STATUS
fi

echo ""
echo "========================================"
echo "Job completed at $(date)"
echo "Total runtime: $SECONDS seconds"
echo "========================================"
