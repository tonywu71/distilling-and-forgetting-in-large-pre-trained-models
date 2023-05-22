# ======= 1. Generate CSV from eval outputs =======

# --- Multilingual ---
# tiny:
python scripts/report_utils/plot_wer_wrt_perplexity/merge_wer_and_ppl_to_csv.py \
    outputs/vanilla/tiny/whisper-tiny-mls.csv \
    outputs/vanilla/tiny/whisper-tiny-implicit_lm-perplexity-mls.csv \
    tiny_multilingual-vanilla-mls

python scripts/report_utils/plot_wer_wrt_perplexity/merge_wer_and_ppl_to_csv.py \
    outputs/finetuning/whisper_tiny-librispeech_clean_100h-benchmark-freeze_encoder/checkpoint-3500-mls.csv \
    outputs/finetuning/whisper_tiny-librispeech_clean_100h-benchmark-freeze_encoder/perplexity/checkpoint-3500-implicit_lm-perplexity-mls.csv \
    tiny_multilingual-finetuned-mls

# medium:
python scripts/report_utils/plot_wer_wrt_perplexity/merge_wer_and_ppl_to_csv.py \
    outputs/vanilla/medium/whisper-medium-mls.csv \
    outputs/vanilla/medium/whisper-medium-implicit_lm-perplexity-mls.csv \
    medium_multilingual-vanilla-mls


# ======= 2. Plot =======

# --- PPL vs WER - tiny multilingual on MLS ---
python scripts/report_utils/plot_wer_wrt_perplexity/plot_wer_wrt_ppl.py \
    outputs/report/plot_wer_wrt_perplexity/mls/wer_and_ppl-tiny_multilingual-vanilla-mls.csv \
    outputs/report/plot_wer_wrt_perplexity/mls/wer_and_ppl-tiny_multilingual-finetuned-mls.csv \
    --kind regression \
    --filename tiny_multilingual_mls
