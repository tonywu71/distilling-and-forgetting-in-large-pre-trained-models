python scripts/report_utils/compare_2_models/compare_2_models_to_latex.py \
    "outputs/vanilla/whisper-tiny-implicit_lm-perplexity-esb.csv" \
    "outputs/finetuning/whisper_tiny-librispeech_clean_100h-benchmark-freeze_encoder/perplexity/checkpoint-3500-implicit_lm-perplexity-esb.csv" \
    --dataset-group esb
