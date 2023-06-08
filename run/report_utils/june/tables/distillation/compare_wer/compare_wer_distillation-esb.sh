# Instructions: Run scripts in batch of 2 and manually concatenate the results in LaTeX.


# ========    Word-level   ========
python scripts/report_utils/compare_wer/compare_wer-2_models/compare_wer-distillation_to_latex_table.py \
    outputs/distillation/whisper_medium_to_tiny/ami_10h/word_level/checkpoint-250-esb_ami.csv \
    --filepath-small-model outputs/vanilla/tiny/whisper-tiny-esb_ami.csv \
    --filepath-large-model outputs/vanilla/medium/whisper-medium-esb_ami.csv \
    --dataset-group esb_ami


python scripts/report_utils/compare_wer/compare_wer-2_models/compare_wer-distillation_to_latex_table.py \
    outputs/distillation/whisper_medium_to_tiny/ami_10h/word_level/checkpoint-250-mls.csv \
    --filepath-small-model outputs/vanilla/tiny/whisper-tiny-mls.csv \
    --filepath-large-model outputs/vanilla/medium/whisper-medium-mls.csv \
    --dataset-group mls


# ========    Sequence-level uniform   ========


# ========    Sequence-level ranked   ========

