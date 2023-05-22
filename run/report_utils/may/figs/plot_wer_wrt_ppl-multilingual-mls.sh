# ======= 1. Generate CSV from eval outputs =======

# --- Multilingual ---
# tiny:
python scripts/report_utils/plot_wer_wrt_perplexity/merge_wer_and_ppl_to_csv.py \
    outputs/vanilla/tiny/whisper-tiny-mls.csv \
    outputs/vanilla/tiny/whisper-tiny-implicit_lm-perplexity-mls.csv \
    tiny_multilingual-vanilla-mls

# tiny finetuned:
python scripts/report_utils/plot_wer_wrt_perplexity/merge_wer_and_ppl_to_csv.py \
    outputs/finetuning/english_model/finetune-whisper_tiny_en-librispeech_clean_100h-freeze_encoder/checkpoint-2500-mls.csv \
    outputs/finetuning/english_model/finetune-whisper_tiny_en-librispeech_clean_100h-freeze_encoder/checkpoint-2500-implicit_lm-perplexity-mls.csv \
    tiny_multilingual-finetuned-mls

# base:
python scripts/report_utils/plot_wer_wrt_perplexity/merge_wer_and_ppl_to_csv.py \
    outputs/vanilla/base/whisper-base-mls.csv \
    outputs/vanilla/base/whisper-base-implicit_lm-perplexity-mls.csv \
    base_multilingual-vanilla-mls

# small:
python scripts/report_utils/plot_wer_wrt_perplexity/merge_wer_and_ppl_to_csv.py \
    outputs/vanilla/small/whisper-small-mls.csv \
    outputs/vanilla/small/whisper-small-implicit_lm-perplexity-mls.csv \
    small_multilingual-vanilla-mls

# medium:
python scripts/report_utils/plot_wer_wrt_perplexity/merge_wer_and_ppl_to_csv.py \
    outputs/vanilla/medium/whisper-medium-mls.csv \
    outputs/vanilla/medium/whisper-medium-implicit_lm-perplexity-mls.csv \
    medium_multilingual-vanilla-mls


# ======= 2. Plot =======

# --- PPL vs WER - comparison of all vanilla Whisper models ---
python scripts/report_utils/plot_wer_wrt_perplexity/plot_wer_wrt_ppl.py \
    outputs/report/plot_wer_wrt_perplexity/mls/wer_and_ppl-tiny_multilingual-vanilla-mls.csv \
    outputs/report/plot_wer_wrt_perplexity/mls/wer_and_ppl-tiny_multilingual-finetuned-mls.csv \
    --kind regression \
    --filename tiny_multilingual_mls

python scripts/report_utils/plot_wer_wrt_perplexity/plot_wer_wrt_ppl.py \
    outputs/report/plot_wer_wrt_perplexity/mls/wer_and_ppl-tiny_multilingual-vanilla-mls.csv \
    outputs/report/plot_wer_wrt_perplexity/mls/wer_and_ppl-base_multilingual-vanilla-mls.csv \
    outputs/report/plot_wer_wrt_perplexity/mls/wer_and_ppl-small_multilingual-vanilla-mls.csv \
    outputs/report/plot_wer_wrt_perplexity/mls/wer_and_ppl-medium_multilingual-vanilla-mls.csv \
    --kind regression \
    --filename all_multilingual_mls
