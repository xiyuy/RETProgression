#!/bin/bash
#SBATCH -J ensemble_predictions
#SBATCH -p short
#SBATCH -N 1
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-user=r.garridogarcia@northeastern.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# Configuration
CONDA_ENV="/projects/retprogression/pytorch_cu121_env"
BASE_DIR="/projects/retprogression/rgarridogarcia"
REPO_DIR="/home/r.garridogarcia/MIGHTE/retprogression/swinV2/RETProgression"

# Output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${BASE_DIR}/ensemble/ensemble_results_${TIMESTAMP}"

# Prediction files from each partition
# MODIFY THESE PATHS to match your actual prediction files
PREDICTIONS=(
    "${BASE_DIR}/gradable_dr_new_5k_clean_test_cropped_centered_wt_allR_1_*/predictions*.csv"
    "${BASE_DIR}/gradable_dr_new_5k_clean_test_cropped_centered_wt_allR_2_*/predictions*.csv"
    "${BASE_DIR}/gradable_dr_new_5k_clean_test_cropped_centered_wt_allR_3_*/predictions*.csv"
    "${BASE_DIR}/gradable_dr_new_5k_clean_test_cropped_centered_wt_allR_4_*/predictions*.csv"
    "${BASE_DIR}/gradable_dr_new_5k_clean_test_cropped_centered_wt_allR_5_*/predictions*.csv"
)

# Ensemble configuration
METHODS="all"  # Try all standard methods: average, logit, median, confidence, max_confidence

# NEW OPTIONS - Uncomment to enable
CONSERVATIVE_VOTING="--conservative_voting"  # Add majority_60, majority_80, unanimous
TUNE_THRESHOLD="--tune_threshold"            # Find optimal thresholds for best method

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

# Start
print_info "Starting ensemble creation at $(date)"
print_info "Configuration:"
print_info "  Methods: ${METHODS}"
print_info "  Conservative voting: ${CONSERVATIVE_VOTING:+ENABLED}"
print_info "  Threshold tuning: ${TUNE_THRESHOLD:+ENABLED}"

# Load modules
module load anaconda3/2024.06 discovery

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

# Change to source directory
cd "${REPO_DIR}/src"
print_info "Working directory: $(pwd)"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Expand wildcards and find actual files
print_info "Finding prediction files..."
ACTUAL_FILES=()
for pattern in "${PREDICTIONS[@]}"; do
    for file in $pattern; do
        if [ -f "$file" ]; then
            ACTUAL_FILES+=("$file")
            print_info "  Found: $(basename $file)"
        fi
    done
done

# Check if we found any files
if [ ${#ACTUAL_FILES[@]} -eq 0 ]; then
    print_error "No prediction files found!"
    print_error "Searched patterns:"
    for pattern in "${PREDICTIONS[@]}"; do
        print_error "  $pattern"
    done
    print_info "\nTip: List your actual prediction files with:"
    print_info "  find ${BASE_DIR} -name 'predictions*.csv' -path '*gradable_dr_new_5k*'"
    exit 1
fi

print_success "Found ${#ACTUAL_FILES[@]} prediction files"

# Verify all files have required columns
print_info "Verifying prediction file formats..."
for file in "${ACTUAL_FILES[@]}"; do
    if ! head -1 "$file" | grep -q "id\|true_label\|prob"; then
        print_warning "File may have unexpected format: $(basename $file)"
        print_warning "Columns: $(head -1 $file)"
    fi
done

# Build ensemble command
ENSEMBLE_CMD="python ensemble_cross_validation.py \
    --predictions ${ACTUAL_FILES[@]} \
    --output_dir ${OUTPUT_DIR} \
    --methods ${METHODS}"

# Add optional flags if enabled
if [ -n "${CONSERVATIVE_VOTING:-}" ]; then
    ENSEMBLE_CMD="${ENSEMBLE_CMD} ${CONSERVATIVE_VOTING}"
    print_info "Will test conservative voting methods (majority_60, majority_80, unanimous)"
fi

if [ -n "${TUNE_THRESHOLD:-}" ]; then
    ENSEMBLE_CMD="${ENSEMBLE_CMD} ${TUNE_THRESHOLD}"
    print_info "Will tune decision threshold on best method"
fi

# Run ensemble
print_info "Running ensemble..."
print_info "Command: ${ENSEMBLE_CMD}"
echo ""

eval ${ENSEMBLE_CMD}

# Check if successful
if [ $? -eq 0 ]; then
    print_success "Ensemble completed successfully!"
    print_info "Results saved to: ${OUTPUT_DIR}"
    
    echo ""
    print_info "="*70
    print_info "GENERATED FILES:"
    print_info "="*70
    ls -lh "${OUTPUT_DIR}"
    
    echo ""
    # Display comparison if available
    if [ -f "${OUTPUT_DIR}/ensemble_comparison.csv" ]; then
        print_info "ENSEMBLE PERFORMANCE COMPARISON:"
        print_info "="*70
        
        # Try to display nicely formatted, fall back to plain cat
        if command -v column &> /dev/null; then
            head -20 "${OUTPUT_DIR}/ensemble_comparison.csv" | column -t -s',' 2>/dev/null || cat "${OUTPUT_DIR}/ensemble_comparison.csv"
        else
            cat "${OUTPUT_DIR}/ensemble_comparison.csv"
        fi
        echo ""
    fi
    
    # Display summary
    if [ -f "${OUTPUT_DIR}/summary.json" ]; then
        print_info "SUMMARY:"
        print_info "="*70
        cat "${OUTPUT_DIR}/summary.json"
        echo ""
    fi
    
    # Highlight if threshold analysis was done
    if [ -f "${OUTPUT_DIR}/threshold_analysis.csv" ]; then
        print_success "Threshold analysis completed!"
        print_info "See detailed results in: threshold_analysis.csv"
        
        # Show key thresholds
        print_info "\nKey Operating Points:"
        python -c "
import pandas as pd
df = pd.read_csv('${OUTPUT_DIR}/threshold_analysis.csv')

# Best balanced accuracy
best_bal = df.loc[df['balanced_accuracy'].idxmax()]
print(f'  Best Balanced Acc (thresh={best_bal[\"threshold\"]:.2f}): '
      f'Sens={best_bal[\"sensitivity\"]:.3f}, Spec={best_bal[\"specificity\"]:.3f}')

# Best specificity with high sensitivity
high_sens = df[df['sensitivity'] >= 0.95]
if len(high_sens) > 0:
    best_spec = high_sens.loc[high_sens['specificity'].idxmax()]
    print(f'  Best Spec+HighSens (thresh={best_spec[\"threshold\"]:.2f}): '
          f'Sens={best_spec[\"sensitivity\"]:.3f}, Spec={best_spec[\"specificity\"]:.3f}')
" 2>/dev/null || print_info "  (See threshold_analysis.csv for details)"
        echo ""
    fi
    
    # Show recommendation
    print_info "="*70
    print_success "NEXT STEPS:"
    print_info "="*70
    print_info "1. Review: ${OUTPUT_DIR}/ensemble_comparison.csv"
    print_info "2. Choose best method based on your needs:"
    print_info "   - Highest balanced accuracy"
    print_info "   - Best sensitivity/specificity trade-off"
    print_info "   - Clinical requirements"
    if [ -f "${OUTPUT_DIR}/threshold_analysis.csv" ]; then
        print_info "3. If needed, apply custom threshold using threshold_analysis.csv"
    fi
    print_info "4. Use predictions from: ensemble_<method>.csv"
    echo ""
    
else
    print_error "Ensemble failed!"
    
    # Show log if available
    LOG_FILE="${OUTPUT_DIR}/ensemble_crossval.log"
    if [ -f "${LOG_FILE}" ]; then
        print_error "Last 30 lines of log:"
        echo "----------------------------------------"
        tail -n 30 "${LOG_FILE}"
        echo "----------------------------------------"
    fi
    
    exit 1
fi

print_success "Script completed at $(date)"
print_info "Full results: ${OUTPUT_DIR}"