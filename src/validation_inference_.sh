#!/bin/bash
#SBATCH -J gradable_dr_new_5k_clean_test_cropped_centered_wt_allR_4 # Job name
#SBATCH -p sharing
#SBATCH -N 1 # Number of nodes
#SBATCH --time=1:00:00  # Inference is much faster than training
#SBATCH --gres=gpu:h100:1
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

# Force single GPU mode
export CUDA_VISIBLE_DEVICES=0
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"

# Configuration - MODIFY THESE PATHS AS NEEDED
PARTITION_ID=1
CHECKPOINT_PATH="/projects/retprogression/rgarridogarcia/checkpoints/1024_Blackcropped_wt_allR_${PARTITION_ID}/best_balanced_acc_model.pth" #cropped1024_cropped_centered_wt_allR_
DATA_DIR="/projects/retprogression/"
IMG_DIR="complete_cropped_1024_centered_wt_10272025_allRight"   # <-- set this to whatever folder you want
ANNOTATIONS_FILE="/gradable_dr/clean_gradable_dr_${PARTITION_ID}_test.csv"
OUTPUT_DIR="/projects/retprogression/rgarridogarcia/gradable_dr_test/gradable_dr_test_cropped_centered_wt_allR_${PARTITION_ID}_${TIMESTAMP}"
BATCH_SIZE=8  # Smaller batch size for inference safety
NUM_WORKERS=4
RESOLUTION=1024
MODEL_NAME="swinv2_large_window12to16_192to256.ms_in22k_ft_in1k"
IMG_SIZE=1024

# Print configuration
echo "========================================"
echo "VALIDATION INFERENCE CONFIGURATION"
echo "========================================"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Data directory: $DATA_DIR"
echo "Annotations file: $ANNOTATIONS_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Resolution: $RESOLUTION"
echo "Model: $MODEL_NAME"
echo "========================================"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint file not found at $CHECKPOINT_PATH"
    echo "Available checkpoints in directory:"
    ls -la $(dirname "$CHECKPOINT_PATH")/ | grep "\.pth$"
    exit 1
fi

# Check if validation annotations file exists
if [ ! -f "$DATA_DIR/$ANNOTATIONS_FILE" ]; then
    echo "ERROR: Validation annotations file not found at $DATA_DIR/$ANNOTATIONS_FILE"
    echo "Available CSV files in data directory:"
    ls -la "$DATA_DIR"/*.csv
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run validation inference
python validation_inference.py \
  --checkpoint "$CHECKPOINT_PATH" \
  --data_dir "$DATA_DIR" \
  --annotations_file "$ANNOTATIONS_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS \
  --resolution $RESOLUTION \
  --model_name "$MODEL_NAME" \
  --img_size $IMG_SIZE \
  --img_dir "$IMG_DIR" \      
  --device cuda

# Check if inference completed successfully
if [ $? -eq 0 ]; then
    echo "========================================"
    echo "VALIDATION INFERENCE COMPLETED SUCCESSFULLY!"
    echo "========================================"
    echo "Output files:"
    ls -la "$OUTPUT_DIR"
    echo ""
    echo "Main results file: $OUTPUT_DIR/validation_detailed_results.csv"
    echo "Summary metrics: $OUTPUT_DIR/validation_summary_metrics.csv"
    echo "Error analysis: $OUTPUT_DIR/validation_errors.csv"
    echo "========================================"
else
    echo "ERROR: Validation inference failed!"
    exit 1
fi