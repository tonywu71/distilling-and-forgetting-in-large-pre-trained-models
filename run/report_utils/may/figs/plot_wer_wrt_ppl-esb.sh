# ======= 1. Generate CSV from eval outputs =======

# --- Multilingual ---
# tiny:
python scripts/report_utils/plot_wer_wrt_perplexity/merge_wer_and_ppl_to_csv.py \
    outputs/vanilla/tiny/whisper-tiny-esb.csv \
    outputs/vanilla/tiny/whisper-tiny-implicit_lm-perplexity-esb.csv \
    tiny_multilingual-vanilla-esb

python scripts/report_utils/plot_wer_wrt_perplexity/merge_wer_and_ppl_to_csv.py \
    outputs/finetuning/whisper_tiny-librispeech_clean_100h-benchmark-freeze_encoder/checkpoint-3500-esb.csv \
    outputs/finetuning/whisper_tiny-librispeech_clean_100h-benchmark-freeze_encoder/perplexity/checkpoint-3500-implicit_lm-perplexity-esb.csv \
    tiny_multilingual-finetuned-esb

# medium:
python scripts/report_utils/plot_wer_wrt_perplexity/merge_wer_and_ppl_to_csv.py \
    outputs/vanilla/medium/whisper-medium-esb.csv \
    outputs/vanilla/medium/whisper-medium-implicit_lm-perplexity-esb.csv \
    medium_multilingual-vanilla-esb


# --- English ---
python scripts/report_utils/plot_wer_wrt_perplexity/merge_wer_and_ppl_to_csv.py \
    outputs/vanilla/tiny-en/whisper-tiny-en-esb.csv \
    outputs/vanilla/tiny-en/whisper-tiny-en-implicit_lm-perplexity-esb.csv \
    tiny_english-vanilla-esb

python scripts/report_utils/plot_wer_wrt_perplexity/merge_wer_and_ppl_to_csv.py \
    outputs/finetuning/english_model/finetune-whisper_tiny_en-librispeech_clean_100h-freeze_encoder/checkpoint-2500-esb.csv \
    outputs/finetuning/english_model/finetune-whisper_tiny_en-librispeech_clean_100h-freeze_encoder/checkpoint-2500-implicit_lm-perplexity-esb.csv \
    tiny_english-finetuned-esb



# ======= 2. Plot =======

# --- All `tiny` ---
python scripts/report_utils/plot_wer_wrt_perplexity/plot_wer_wrt_ppl.py \
    outputs/report/plot_wer_wrt_perplexity/esb/wer_and_ppl-tiny_multilingual-vanilla-esb.csv \
    outputs/report/plot_wer_wrt_perplexity/esb/wer_and_ppl-tiny_multilingual-finetuned-esb.csv \
    outputs/report/plot_wer_wrt_perplexity/esb/wer_and_ppl-tiny_english-vanilla-esb.csv \
    outputs/report/plot_wer_wrt_perplexity/esb/wer_and_ppl-tiny_english-finetuned-esb.csv \
    --filename all-tiny-esb

python scripts/report_utils/plot_wer_wrt_perplexity/plot_wer_wrt_ppl.py \
    outputs/report/plot_wer_wrt_perplexity/esb/wer_and_ppl-tiny_multilingual-vanilla-esb.csv \
    outputs/report/plot_wer_wrt_perplexity/esb/wer_and_ppl-tiny_multilingual-finetuned-esb.csv \
    outputs/report/plot_wer_wrt_perplexity/esb/wer_and_ppl-tiny_english-vanilla-esb.csv \
    outputs/report/plot_wer_wrt_perplexity/esb/wer_and_ppl-tiny_english-finetuned-esb.csv \
    outputs/report/plot_wer_wrt_perplexity/wer_and_ppl-medium_multilingual-vanilla-esb.csv \
    --kind regression \
    --filename all-tiny-esb


# --- Multilingual ---
python scripts/report_utils/plot_wer_wrt_perplexity/plot_wer_wrt_ppl.py \
    outputs/report/plot_wer_wrt_perplexity/esb/wer_and_ppl-tiny_multilingual-vanilla-esb.csv \
    outputs/report/plot_wer_wrt_perplexity/esb/wer_and_ppl-tiny_multilingual-finetuned-esb.csv \
    --filename multilingual-esb

python scripts/report_utils/plot_wer_wrt_perplexity/plot_wer_wrt_ppl.py \
    outputs/report/plot_wer_wrt_perplexity/esb/wer_and_ppl-tiny_multilingual-vanilla-esb.csv \
    outputs/report/plot_wer_wrt_perplexity/esb/wer_and_ppl-tiny_multilingual-finetuned-esb.csv \
    --kind regression \
    --filename multilingual-esb


# --- English ---
python scripts/report_utils/plot_wer_wrt_perplexity/plot_wer_wrt_ppl.py \
    outputs/report/plot_wer_wrt_perplexity/esb/wer_and_ppl-tiny_english-vanilla-esb.csv \
    outputs/report/plot_wer_wrt_perplexity/esb/wer_and_ppl-tiny_english-finetuned-esb.csv \
    --filename english-esb

python scripts/report_utils/plot_wer_wrt_perplexity/plot_wer_wrt_ppl.py \
    outputs/report/plot_wer_wrt_perplexity/esb/wer_and_ppl-tiny_english-vanilla-esb.csv \
    outputs/report/plot_wer_wrt_perplexity/esb/wer_and_ppl-tiny_english-finetuned-esb.csv \
    --kind regression \
    --filename english-esb


# --- tiny vs medium ---
python scripts/report_utils/plot_wer_wrt_perplexity/plot_wer_wrt_ppl.py \
    outputs/report/plot_wer_wrt_perplexity/esb/wer_and_ppl-tiny_multilingual-vanilla-esb.csv \
    outputs/report/plot_wer_wrt_perplexity/esb/wer_and_ppl-medium_multilingual-vanilla-esb.csv \
    --filename tiny_vs_medium-esb

python scripts/report_utils/plot_wer_wrt_perplexity/plot_wer_wrt_ppl.py \
    outputs/report/plot_wer_wrt_perplexity/esb/wer_and_ppl-tiny_multilingual-vanilla-esb.csv \
    outputs/report/plot_wer_wrt_perplexity/esb/wer_and_ppl-medium_multilingual-vanilla-esb.csv \
    --kind regression \
    --filename tiny_vs_medium-esb
