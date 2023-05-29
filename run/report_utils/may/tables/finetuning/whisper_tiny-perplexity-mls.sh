python scripts/report_utils/compare_2_models/compare_wer/compare_wer-2_models_to_latex.py \
    "outputs/vanilla/tiny/whisper-tiny-implicit_lm-perplexity-mls.csv" \
    "outputs/finetuning/whisper_tiny-librispeech_clean_100h-benchmark-freeze_encoder/perplexity/checkpoint-3500-implicit_lm-perplexity-mls.csv" \
    --dataset-group mls
