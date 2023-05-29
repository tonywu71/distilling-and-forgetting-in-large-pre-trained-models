# Generate CSV from eval outputs:
python scripts/report_utils/compare_wer/compare_wer-multiple_models_to_csv.py \
	outputs/finetuning/whisper_tiny-librispeech_clean_100h-benchmark-freeze_encoder/perplexity/checkpoint-500-implicit_lm-perplexity-fab.csv \
	outputs/finetuning/whisper_tiny-librispeech_clean_100h-benchmark-freeze_encoder/perplexity/checkpoint-1000-implicit_lm-perplexity-fab.csv \
	outputs/finetuning/whisper_tiny-librispeech_clean_100h-benchmark-freeze_encoder/perplexity/checkpoint-1500-implicit_lm-perplexity-fab.csv \
    outputs/finetuning/whisper_tiny-librispeech_clean_100h-benchmark-freeze_encoder/perplexity/checkpoint-2000-implicit_lm-perplexity-fab.csv \
    outputs/finetuning/whisper_tiny-librispeech_clean_100h-benchmark-freeze_encoder/perplexity/checkpoint-2500-implicit_lm-perplexity-fab.csv \
    outputs/finetuning/whisper_tiny-librispeech_clean_100h-benchmark-freeze_encoder/perplexity/checkpoint-3000-implicit_lm-perplexity-fab.csv \
    outputs/finetuning/whisper_tiny-librispeech_clean_100h-benchmark-freeze_encoder/perplexity/checkpoint-3500-implicit_lm-perplexity-fab.csv


# Generate plot from previous CSV:
python scripts/report_utils/plot_forgetting_perplexity_wrt_checkpoints_on_fab.py \
    outputs/report/compare_perplexity_multiple_models/perplexity_wrt_checkpoints-fab.csv

python scripts/report_utils/plot_forgetting_perplexity_wrt_checkpoints_on_fab.py \
    outputs/report/compare_perplexity_multiple_models/perplexity_wrt_checkpoints-fab.csv --is-relative
