python scripts/report_utils/compare_2_models/compare_2_models_to_latex.py \
    "outputs/finetuning/vanilla/whisper-tiny-esb.csv" \
    "outputs/finetuning/whisper_tiny-librispeech_clean_100h-benchmark-freeze_encoder/checkpoint-3500-esb.csv" \
    --dataset-group esb
