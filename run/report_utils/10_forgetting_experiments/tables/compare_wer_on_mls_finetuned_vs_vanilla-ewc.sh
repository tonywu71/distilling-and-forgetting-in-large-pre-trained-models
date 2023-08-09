python scripts/report_utils/compare_wer/compare_wer-2_models-to_latex.py \
    outputs/vanilla/tiny/whisper-tiny-mlsdc.csv \
    outputs/finetune_ewc/whisper_tiny/preserve_french/ami_100h-lambda_1e-4/checkpoint-3000-mlsdc-normalized.csv \
    --dataset-group mls


python scripts/report_utils/compare_wer/compare_wer-multiple_models-to_latex.py \
    outputs/vanilla/tiny/whisper-tiny-mlsdc.csv \
    outputs/finetuning/whisper_tiny/ami_100h/checkpoint-3000-mlsdc-normalized.csv \
    outputs/finetune_ewc/whisper_tiny/preserve_french/ami_100h-lambda_1e-4/checkpoint-3000-mlsdc-normalized.csv
