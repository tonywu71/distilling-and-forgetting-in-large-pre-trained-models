python scripts/report_utils/compare_2_models/compare_2_models_to_latex.py \
    "outputs/finetuning/vanilla/whisper-tiny-implicit_lm-perplexity-mls.csv" \
    "outputs/finetuning/whisper_tiny-librispeech_clean_100h-benchmark-freeze_encoder/perplexity/checkpoint-3500-implicit_lm-perplexity-mls.csv" \
    --dataset-group mls
