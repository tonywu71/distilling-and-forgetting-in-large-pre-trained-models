python scripts/report_utils/compare_wer/compare_wer-2_models/compare_wer-2_models_to_latex.py \
    outputs/vanilla/tiny/whisper-tiny-esbdc.csv \
    outputs/finetuning/whisper_tiny/compare_freezing_strategies_on_librispeech_clean/librispeech_clean_100h-benchmark-freeze_encoder/forgetting/freeze_encoder_final-esbdc.csv \
    --dataset-group esb_librispeech

python scripts/report_utils/compare_wer/compare_wer-2_models/compare_wer-2_models_to_latex.py \
    outputs/vanilla/tiny/whisper-tiny-esbdc.csv \
    outputs/finetuning/whisper_tiny/ami_100h/finetuned_on_ami-esbdc.csv \
    --dataset-group esb_ami
