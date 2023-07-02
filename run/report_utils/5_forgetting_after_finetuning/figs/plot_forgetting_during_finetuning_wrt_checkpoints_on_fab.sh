python scripts/report_utils/forgetting/concat_csv_wer_wrt_steps.py \
    outputs/finetuning/whisper_tiny/ami_100h/forgetting_wrt_steps_on_fab

python scripts/report_utils/forgetting/plot_forgetting_wer_wrt_steps_on_fab.py \
    outputs/finetuning/whisper_tiny/ami_100h/wer_wrt_steps_on_fab.csv

python scripts/report_utils/forgetting/plot_forgetting_wer_wrt_steps_on_fab.py \
    outputs/finetuning/whisper_tiny/ami_100h/wer_wrt_steps_on_fab.csv \
    --is-relative
