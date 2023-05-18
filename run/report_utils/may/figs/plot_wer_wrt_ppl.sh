# ======= 1. Generate CSV from eval outputs =======
# --- Multilingual ---
python scripts/report_utils/plot_wer_wrt_perplexity/merge_wer_and_ppl_to_csv.py \
    outputs/vanilla/whisper-tiny-esb.csv \
    outputs/vanilla/whisper-tiny-implicit_lm-perplexity-esb.csv \
    tiny_multilingual-vanilla

python scripts/report_utils/plot_wer_wrt_perplexity/merge_wer_and_ppl_to_csv.py \
    outputs/finetuning/whisper_tiny-librispeech_clean_100h-benchmark-freeze_encoder/checkpoint-3500-esb.csv \
    outputs/finetuning/whisper_tiny-librispeech_clean_100h-benchmark-freeze_encoder/perplexity/checkpoint-3500-implicit_lm-perplexity-esb.csv \
    tiny_multilingual-finetuned

# --- English ---
python scripts/report_utils/plot_wer_wrt_perplexity/merge_wer_and_ppl_to_csv.py \
    outputs/vanilla/whisper-tiny-en-esb.csv \
    outputs/vanilla/whisper-tiny-en-implicit_lm-perplexity-esb.csv \
    tiny_english-vanilla

python scripts/report_utils/plot_wer_wrt_perplexity/merge_wer_and_ppl_to_csv.py \
    ?? \
    ?? \
    tiny_english-finetuned


# ======= 2. Plot =======
python scripts/report_utils/plot_wer_wrt_perplexity/plot_wer_wrt_ppl.py \
    outputs/report/plot_wer_wrt_perplexity/wer_and_ppl-tiny_english-vanilla.csv \
    outputs/report/plot_wer_wrt_perplexity/wer_and_ppl-tiny_multilingual-finetuned.csv \
    outputs/report/plot_wer_wrt_perplexity/wer_and_ppl-tiny_multilingual-vanilla.csv \
    --filename all

python scripts/report_utils/plot_wer_wrt_perplexity/plot_wer_wrt_ppl.py \
    outputs/report/plot_wer_wrt_perplexity/wer_and_ppl-tiny_english-vanilla.csv \
    outputs/report/plot_wer_wrt_perplexity/wer_and_ppl-tiny_multilingual-finetuned.csv \
    outputs/report/plot_wer_wrt_perplexity/wer_and_ppl-tiny_multilingual-vanilla.csv \
    --regression \
    --filename all

python scripts/report_utils/plot_wer_wrt_perplexity/plot_wer_wrt_ppl.py \
    outputs/report/plot_wer_wrt_perplexity/wer_and_ppl-tiny_multilingual-finetuned.csv \
    outputs/report/plot_wer_wrt_perplexity/wer_and_ppl-tiny_multilingual-vanilla.csv \
    --filename multilingual

python scripts/report_utils/plot_wer_wrt_perplexity/plot_wer_wrt_ppl.py \
    outputs/report/plot_wer_wrt_perplexity/wer_and_ppl-tiny_multilingual-finetuned.csv \
    outputs/report/plot_wer_wrt_perplexity/wer_and_ppl-tiny_multilingual-vanilla.csv \
    --regression \
    --filename multilingual
