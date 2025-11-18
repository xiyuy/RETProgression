#!/bin/bash
set -euo pipefail

# temporarily disable nounset while sourcing bashrc (it references unset vars)
set +u
source ~/.bashrc || true
set -u

conda activate /projects/retprogression/pytorch_cu121_env
python test_inference.py \
  --checkpoint "/projects/retprogression/rgarridogarcia/checkpoints/clean_gradable_swinv2_resolution1024_oversample50_RetinaCropped_BrightnessFilter/best_balanced_acc_model.pth" \
  --data_dir /projects/retprogression \
  --annotations_file complete_gradable_dr_test_PRESENT.csv \
  --img_dir complete_cropped1024_brightness_09112025 \
  --output_dir /projects/retprogression/rgarridogarcia/complete_gradable_swinv2_resolution1024_oversample50_RetinaCropped_BrightnessFilter \
  --model_name swinv2_large_window12to16_192to256.ms_in22k_ft_in1k \
  --img_size 1024 \
  --batch_size 32 \
  --num_workers 4


# to run
# bash run_test.sh

