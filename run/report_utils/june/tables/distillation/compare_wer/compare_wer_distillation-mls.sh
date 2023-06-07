# Instructions: Run scripts in batch of 2 and manually concatenate the results in LaTeX.

python scripts/report_utils/compare_wer/compare_wer-2_models/compare_wer-2_models_to_latex.py \
    outputs/vanilla/tiny/whisper-tiny-mls.csv \
    TOFILL \
    --dataset-group mls
python scripts/report_utils/compare_wer/compare_wer-2_models/compare_wer-2_models_to_latex.py \
    outputs/vanilla/medium/whisper-medium-mls.csv \
    TOFILL \
    --dataset-group mls


# ========    Sequence-level uniform   ========


# ========    Sequence-level ranked   ========

