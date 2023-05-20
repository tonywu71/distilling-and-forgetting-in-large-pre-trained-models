# ======= 1. Generate CSV from eval outputs =======

# --- Multilingual ---
python scripts/report_utils/plot_wer_wrt_perplexity/merge_wer_and_ppl_to_csv.py \
    outputs/vanilla/tiny/whisper-tiny-mls.csv \
    outputs/vanilla/tiny/whisper-tiny-implicit_lm-perplexity-mls.csv \
    tiny_multilingual-vanilla-mls

python scripts/report_utils/plot_wer_wrt_perplexity/merge_wer_and_ppl_to_csv.py \
    outputs/finetuning/whisper_tiny-librispeech_clean_100h-benchmark-freeze_encoder/checkpoint-3500-mls.csv \
    outputs/finetuning/whisper_tiny-librispeech_clean_100h-benchmark-freeze_encoder/perplexity/checkpoint-3500-implicit_lm-perplexity-mls.csv \
    tiny_multilingual-finetuned-mls


# --- English ---
python scripts/report_utils/plot_wer_wrt_perplexity/merge_wer_and_ppl_to_csv.py \
    outputs/vanilla/tiny-en/whisper-tiny-en-mls.csv \
    outputs/vanilla/tiny-en/whisper-tiny-en-implicit_lm-perplexity-mls.csv \
    tiny_english-vanilla-mls

python scripts/report_utils/plot_wer_wrt_perplexity/merge_wer_and_ppl_to_csv.py \
    outputs/finetuning/english_model/finetune-whisper_tiny_en-librispeech_clean_100h-freeze_encoder/checkpoint-2500-mls.csv \
    outputs/finetuning/english_model/finetune-whisper_tiny_en-librispeech_clean_100h-freeze_encoder/checkpoint-2500-implicit_lm-perplexity-mls.csv \
    tiny_english-finetuned-mls



# ======= 2. Plot =======

# --- All ---
python scripts/report_utils/plot_wer_wrt_perplexity/plot_wer_wrt_ppl.py \
    outputs/report/plot_wer_wrt_perplexity/wer_and_ppl-tiny_multilingual-vanilla-mls.csv \
    outputs/report/plot_wer_wrt_perplexity/wer_and_ppl-tiny_multilingual-finetuned-mls.csv \
    outputs/report/plot_wer_wrt_perplexity/wer_and_ppl-tiny_english-vanilla-mls.csv \
    outputs/report/plot_wer_wrt_perplexity/wer_and_ppl-tiny_english-finetuned-mls.csv \
    --filename all-mls

python scripts/report_utils/plot_wer_wrt_perplexity/plot_wer_wrt_ppl.py \
    outputs/report/plot_wer_wrt_perplexity/wer_and_ppl-tiny_multilingual-vanilla-mls.csv \
    outputs/report/plot_wer_wrt_perplexity/wer_and_ppl-tiny_multilingual-finetuned-mls.csv \
    outputs/report/plot_wer_wrt_perplexity/wer_and_ppl-tiny_english-vanilla-mls.csv \
    outputs/report/plot_wer_wrt_perplexity/wer_and_ppl-tiny_english-finetuned-mls.csv \
    --kind regression \
    --filename all-mls


# --- Multilingual ---
python scripts/report_utils/plot_wer_wrt_perplexity/plot_wer_wrt_ppl.py \
    outputs/report/plot_wer_wrt_perplexity/wer_and_ppl-tiny_multilingual-vanilla-mls.csv \
    outputs/report/plot_wer_wrt_perplexity/wer_and_ppl-tiny_multilingual-finetuned-mls.csv \
    --filename multilingual-mls

python scripts/report_utils/plot_wer_wrt_perplexity/plot_wer_wrt_ppl.py \
    outputs/report/plot_wer_wrt_perplexity/wer_and_ppl-tiny_multilingual-vanilla-mls.csv \
    outputs/report/plot_wer_wrt_perplexity/wer_and_ppl-tiny_multilingual-finetuned-mls.csv \
    --kind regression \
    --filename multilingual-mls


# --- English ---
python scripts/report_utils/plot_wer_wrt_perplexity/plot_wer_wrt_ppl.py \
    outputs/report/plot_wer_wrt_perplexity/wer_and_ppl-tiny_english-vanilla-mls.csv \
    outputs/report/plot_wer_wrt_perplexity/wer_and_ppl-tiny_english-finetuned-mls.csv \
    --filename english-mls

python scripts/report_utils/plot_wer_wrt_perplexity/plot_wer_wrt_ppl.py \
    outputs/report/plot_wer_wrt_perplexity/wer_and_ppl-tiny_english-vanilla-mls.csv \
    outputs/report/plot_wer_wrt_perplexity/wer_and_ppl-tiny_english-finetuned-mls.csv \
    --kind regression \
    --filename english-mls
