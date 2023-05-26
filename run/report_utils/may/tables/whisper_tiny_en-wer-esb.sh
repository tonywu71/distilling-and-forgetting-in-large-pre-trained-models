python scripts/report_utils/compare_2_models/compare_2_models_to_latex.py \
    outputs/vanilla/tiny-en/whisper-tiny-en-esb.csv \
    outputs/finetuning/english_model/finetune-whisper_tiny_en-librispeech_clean_100h-freeze_encoder/checkpoint-2500-esb.csv \
    --dataset-group esb
