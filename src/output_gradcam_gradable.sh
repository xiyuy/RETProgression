#!/usr/bin/env bash
#SBATCH --job-name=saliency_gradable
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-user=r.garridogarcia@northeastern.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# ── Environment ───────────────────────────────────────────────────────────────
CONDA_ENV="/projects/retprogression/pytorch_cu121_env_copy"
BASE_DIR="/projects/retprogression"
REPO_DIR="/home/r.garridogarcia/MIGHTE/retprogression/swinV2/RETProgression"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

module load anaconda3/2024.06 cuda/12.1.1 discovery
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

cd "${REPO_DIR}/src"

# ── Settings ──────────────────────────────────────────────────────────────────
MODEL_NAME="swinv2_large_window12to16_192to256.ms_in22k_ft_in1k"
IMG_DIR="complete_cropped_1024_centered_wt_10272025_allRight"
RAW_IMG_DIR="clean_dataset_07012025_allRight"   # ← set this to your raw images directory
PARTITION_ID=1   # ← change this to run a different partition

CHECKPOINT="${BASE_DIR}/rgarridogarcia/checkpoints/cropped1024_cropped_centered_wt_allR_${PARTITION_ID}/best_balanced_acc_model.pth"
CSV_TEST="${BASE_DIR}/gradable_dr/clean_gradable_dr_1_val.csv"
OUTDIR="${BASE_DIR}/rgarridogarcia/results/smap_gradable/partition_${PARTITION_ID}"

# ── Sanity checks ─────────────────────────────────────────────────────────────
echo "[INFO] PARTITION_ID : ${PARTITION_ID}"
echo "[INFO] Checkpoint   : ${CHECKPOINT}"
echo "[INFO] CSV          : ${CSV_TEST}"
echo "[INFO] Output       : ${OUTDIR}"

[ -f "${CHECKPOINT}" ] || { echo "[ERROR] Checkpoint not found: ${CHECKPOINT}"; exit 1; }
[ -f "${CSV_TEST}"   ] || { echo "[ERROR] CSV not found: ${CSV_TEST}";           exit 1; }

# ── Run ───────────────────────────────────────────────────────────────────────
python gradcam_swinv2_gradable.py \
  --images-dir "${BASE_DIR}/${IMG_DIR}" \
  --csv-test   "${CSV_TEST}" \
  --mode selected \
  --ids "599316 594460 505452 522999 499351 542681 542002 402978 486893 561786" \
  --model      "${MODEL_NAME}" \
  --ckpt       "${CHECKPOINT}" \
  --img-size   1024 \
  --outdir     "${OUTDIR}" \
  --raw-images-dir "${BASE_DIR}/${RAW_IMG_DIR}" \
  --target-source trained

# ── To run ALL test images instead of selected IDs, replace --mode block with:
#   --mode all \
# TP - "381203 401683 401681 376607 581375 334817 50791"
# FP - "495586 531035 322748 507804 496609 532135 438691 385451 385448 502179"
# FN - "531305 531306 603905 572344 603800 594133"
# TN - "599316 594460 505452 522999 499351 542681 542002 402978 486893 561786"