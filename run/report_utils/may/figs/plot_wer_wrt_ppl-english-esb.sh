# ======= 1. Generate CSV from eval outputs =======


# --- English models ---
# tiny:
python scripts/report_utils/plot_wer_wrt_perplexity/merge_wer_and_ppl_to_csv.py \
    outputs/vanilla/tiny-en/whisper-tiny-en-esb.csv \
    outputs/vanilla/tiny-en/whisper-tiny-en-implicit_lm-perplexity-esb.csv \
    tiny_english-vanilla-esb

# tiny finetuned:
python scripts/report_utils/plot_wer_wrt_perplexity/merge_wer_and_ppl_to_csv.py \
    outputs/finetuning/english_model/finetune-whisper_tiny_en-librispeech_clean_100h-freeze_encoder/checkpoint-2500-esb.csv \
    outputs/finetuning/english_model/finetune-whisper_tiny_en-librispeech_clean_100h-freeze_encoder/checkpoint-2500-implicit_lm-perplexity-esb.csv \
    tiny_english-finetuned-esb

# base:
python scripts/report_utils/plot_wer_wrt_perplexity/merge_wer_and_ppl_to_csv.py \
    outputs/vanilla/base-en/whisper-base-en-esb.csv \
    outputs/vanilla/base-en/whisper-base-en-implicit_lm-perplexity-esb.csv \
    base_english-vanilla-esb

# small:
python scripts/report_utils/plot_wer_wrt_perplexity/merge_wer_and_ppl_to_csv.py \
    outputs/vanilla/small-en/whisper-small-en-esb.csv \
    outputs/vanilla/small-en/whisper-small-en-implicit_lm-perplexity-esb.csv \
    small_english-vanilla-esb

# medium:
python scripts/report_utils/plot_wer_wrt_perplexity/merge_wer_and_ppl_to_csv.py \
    outputs/vanilla/medium-en/whisper-medium-en-esb.csv \
    outputs/vanilla/medium-en/whisper-medium-en-implicit_lm-perplexity-esb.csv \
    medium_english-vanilla-esb



# ======= 2. Plot =======

# --- PPL vs WER - impact of fine-tuning on Whisper tiny-en ---
python scripts/report_utils/plot_wer_wrt_perplexity/plot_wer_wrt_ppl.py \
    outputs/report/plot_wer_wrt_perplexity/esb/wer_and_ppl-tiny_english-vanilla-esb.csv \
    outputs/report/plot_wer_wrt_perplexity/esb/wer_and_ppl-tiny_english-finetuned-esb.csv \
    --kind regression \
    --xlim 0 \
    --xlim 16000 \
    --ylim 0 \
    --ylim 50 \
    --filename tiny_english_esb

python scripts/report_utils/plot_wer_wrt_perplexity/plot_wer_wrt_ppl.py \
    outputs/report/plot_wer_wrt_perplexity/esb/wer_and_ppl-tiny_english-vanilla-esb.csv \
    outputs/report/plot_wer_wrt_perplexity/esb/wer_and_ppl-base_english-vanilla-esb.csv \
    outputs/report/plot_wer_wrt_perplexity/esb/wer_and_ppl-small_english-vanilla-esb.csv \
    outputs/report/plot_wer_wrt_perplexity/esb/wer_and_ppl-medium_english-vanilla-esb.csv \
    --kind regression \
    --filename all_english_esb
