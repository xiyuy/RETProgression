#!/bin/bash
# validation_inference.sh - Run model predictions with robust error handling

#SBATCH -J cropped1024_cropped_centered_wt_allR_1       # Job name
#SBATCH -p gpu                         # GPU partition
#SBATCH -N 1                           # Single node
#SBATCH --gres=gpu:v100-sxm2:1              # Request one H100 GPU (adjust name if different)
#SBATCH --cpus-per-task=8              # Number of CPU threads for data loading
#SBATCH --mem=6G                     # System memory (increase if needed)
#SBATCH --time=4:00:00                 # Max runtime (extendable if large dataset)
#SBATCH --output=slurm-%j.out          # Standard output log
#SBATCH --error=slurm-%j.err           # Standard error log
#SBATCH --mail-user=r.garridogarcia@northeastern.edu
#SBATCH --mail-type=END,FAIL           # Notify on job end or failure

set -euo pipefail

# Configuration
CONDA_ENV="/projects/retprogression/pytorch_cu121_env"
BASE_DIR="/projects/retprogression"
REPO_DIR="/home/r.garridogarcia/MIGHTE/retprogression/swinV2/RETProgression"

# Add repo to Python path
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

# Model settings
MODEL_NAME="swinv2_large_window12to16_192to256.ms_in22k_ft_in1k"
IMG_SIZE=1024
BATCH_SIZE=4
NUM_WORKERS=0  # Set to 0 to avoid multiprocessing issues

# Dataset settings
DATA_DIR="${BASE_DIR}"
PARTITION_ID=5
ANNOTATIONS_FILE="gradable_dr/clean_gradable_dr_${PARTITION_ID}_test.csv" #"estenda-ihs-test-set_ungradable.csv"  #gradable_dr/clean_gradable_dr_${PARTITION_ID}_test.csv #"gradable_dr_test800igames.csv" #"gradable_dr_new_5k_clean_test.csv"
IMG_DIR="complete_cropped_1024_centered_wt_10272025_allRight" #"test_cropped1024_wt_10272025" #"estenda-ihs-test-set_centered_wt" #"test_dataset_10032025_blackcropped_wt_allRight" #"test_cropped1024_wt_10272025_allRight" #"complete_cropped_1024_centered_wt_10272025_allRight" #"joslin_centered_1024_wt_12012025"

# Checkpoint settings
CHECKPOINT_DIR="${BASE_DIR}/rgarridogarcia/checkpoints"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/cropped1024_cropped_centered_wt_allR_${PARTITION_ID}/best_balanced_acc_model.pth" #cropped1024_cropped_centered_wt_allR_ #1024_Blackcropped_wt_allR_

# Output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${BASE_DIR}/rgarridogarcia/rerun_5k_best_models/test_cropped1024_cropped_centered_wt_allR_${PARTITION_ID}" #gradable_dr_800images_cropped_centered_wt_allR_${PARTITION_ID}" #cropped_centered_wt_allR_

# Function to print colored messages
print_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

# Start execution
print_info "Starting robust prediction run at $(date)"
print_info "Configuration:"
print_info "  Checkpoint: ${CHECKPOINT_PATH}"
print_info "  Data directory: ${DATA_DIR}"
print_info "  Annotations: ${ANNOTATIONS_FILE}"
print_info "  Image directory: ${IMG_DIR}"
print_info "  Output directory: ${OUTPUT_DIR}"
print_info "  Model: ${MODEL_NAME}"
print_info "  Image size: ${IMG_SIZE}"
print_info "  Batch size: ${BATCH_SIZE}"
print_info "  Workers: ${NUM_WORKERS}"

# Check if checkpoint exists
if [ ! -f "${CHECKPOINT_PATH}" ]; then
    print_error "Checkpoint not found: ${CHECKPOINT_PATH}"
    exit 1
fi

# Check if annotations file exists
if [ ! -f "${DATA_DIR}/${ANNOTATIONS_FILE}" ]; then
    print_error "Annotations file not found: ${DATA_DIR}/${ANNOTATIONS_FILE}"
    exit 1
fi

# Check if image directory exists
if [ ! -d "${DATA_DIR}/${IMG_DIR}" ]; then
    print_warning "Image directory not found: ${DATA_DIR}/${IMG_DIR}"
    print_warning "The script will handle missing files, but this may indicate a problem"
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Save configuration to output directory
CONFIG_FILE="${OUTPUT_DIR}/config.txt"
cat > "${CONFIG_FILE}" << EOF
Prediction Configuration
========================
Date: $(date)
Checkpoint: ${CHECKPOINT_PATH}
Data Directory: ${DATA_DIR}
Annotations File: ${ANNOTATIONS_FILE}
Image Directory: ${IMG_DIR}
Model: ${MODEL_NAME}
Image Size: ${IMG_SIZE}
Batch Size: ${BATCH_SIZE}
Workers: ${NUM_WORKERS}
Output Directory: ${OUTPUT_DIR}
EOF

print_info "Configuration saved to ${CONFIG_FILE}"

# Activate conda environment
print_info "Activating conda environment..."
set +u  # Temporarily disable unset variable check for conda
source ~/.bashrc || true
conda activate "${CONDA_ENV}"
set -u

# Check CUDA availability
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    DEVICE="cuda"
    print_info "CUDA is available, using GPU"
else
    DEVICE="cpu"
    print_warning "CUDA not available, using CPU (this will be slow)"
fi

# Run predictions
print_info "Running robust predictions..."
python validation_inference.py \
    --checkpoint "${CHECKPOINT_PATH}" \
    --model_name "${MODEL_NAME}" \
    --data_dir "${DATA_DIR}" \
    --annotations_file "${ANNOTATIONS_FILE}" \
    --img_dir "${IMG_DIR}" \
    --img_size ${IMG_SIZE} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --output_dir "${OUTPUT_DIR}" \
    --device ${DEVICE} \

# Check if prediction was successful
if [ $? -eq 0 ]; then
    print_success "Predictions completed successfully!"
    print_info "Results saved to: ${OUTPUT_DIR}"
    
    # List output files
    print_info "Generated files:"
    ls -lh "${OUTPUT_DIR}/"*.csv 2>/dev/null || print_warning "No CSV files generated"
    
    # Check for predictions file
    PRED_FILES="${OUTPUT_DIR}/predictions_*.csv"
    for PRED_FILE in ${PRED_FILES}; do
        if [ -f "${PRED_FILE}" ]; then
            NUM_ROWS=$(wc -l < "${PRED_FILE}")
            print_info "Predictions file: $(basename ${PRED_FILE}) (${NUM_ROWS} lines)"
            
            # Show preview
            print_info "Preview of predictions (first 5 rows):"
            head -n 6 "${PRED_FILE}" | column -t -s',' | head -20
            
            # Check for errors file
            ERROR_FILE="${OUTPUT_DIR}/errors_$(basename ${PRED_FILE} | sed 's/predictions_//')"
            if [ -f "${ERROR_FILE}" ]; then
                NUM_ERRORS=$(wc -l < "${ERROR_FILE}")
                print_warning "Found ${NUM_ERRORS} prediction errors. See: $(basename ${ERROR_FILE})"
            fi
        fi
    done
    
    # Check for metrics file
    METRICS_FILES="${OUTPUT_DIR}/metrics_*.csv"
    for METRICS_FILE in ${METRICS_FILES}; do
        if [ -f "${METRICS_FILE}" ]; then
            print_info "Metrics summary:"
            cat "${METRICS_FILE}" | python -c "
import sys
import pandas as pd
df = pd.read_csv(sys.stdin)
for col in df.columns:
    if col not in ['total_samples', 'TP', 'FP', 'FN', 'TN']:
        print(f'  {col}: {df[col].iloc[0]:.4f}')
"
        fi
    done
else
    print_error "Prediction failed. Check the log file in ${OUTPUT_DIR}"
    
    # Show last few lines of log file if it exists
    LOG_FILE="${OUTPUT_DIR}/predictions.log"
    if [ -f "${LOG_FILE}" ]; then
        print_error "Last 20 lines of log file:"
        tail -n 20 "${LOG_FILE}"
    fi
    
    exit 1
fi

print_success "Script completed at $(date)"
print_info "All results are in: ${OUTPUT_DIR}"