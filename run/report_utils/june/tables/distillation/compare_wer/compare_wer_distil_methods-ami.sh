python scripts/report_utils/compare_wer/compare_wer-multiple_models/compare_wer-multiple_models_1_dataset.py \
    outputs/vanilla/tiny/whisper-tiny-esb_ami.csv \
    outputs/vanilla/medium/whisper-medium-esb_ami.csv \
    checkpoints/distillation/whisper_medium_to_tiny/ami_10h/word_level/checkpoint-250 \
    # TODO: Insert checkpoints for the remaining models
    --dataset ami
