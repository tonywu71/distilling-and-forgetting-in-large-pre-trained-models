python scripts/report_utils/compare_wer/compare_wer-2_models/compare_wer-2_models_to_latex.py \
    outputs/vanilla/tiny/whisper-tiny-esb_ami.csv \
    outputs/distillation/whisper_medium_to_tiny/ami_10h/seq_level_1_best-k_1/checkpoint-50-esb_ami.csv \
    --dataset-group esb_ami

# TODO: Manually add a third column after RER for `medium (vanilla)`
