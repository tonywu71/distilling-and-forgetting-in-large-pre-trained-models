python scripts/report_utils/forgetting/concat_csv_wer_wrt_steps.py \
    outputs/finetune_ewc/whisper_tiny/combined/ami_100h/norm_only

python scripts/report_utils/forgetting/plot_forgetting_wer_wrt_steps_on_fab.py \
    outputs/finetune_ewc/whisper_tiny/combined/ami_100h/wer_wrt_steps_on_fab.csv

python scripts/report_utils/forgetting/plot_forgetting_wer_wrt_steps_on_fab.py \
    outputs/finetune_ewc/whisper_tiny/combined/ami_100h/wer_wrt_steps_on_fab.csv \
    --is-relative

python scripts/report_utils/forgetting/plot_forgetting_wer_wrt_steps_on_fab-ewc.py \
    outputs/finetune_ewc/whisper_tiny/combined/ami_100h/wer_wrt_steps_on_fab.csv \
    outputs/finetuning/whisper_tiny/ami_100h/wer_wrt_steps_on_fab.csv

python scripts/report_utils/forgetting/plot_forgetting_wer_wrt_steps_on_fab-ewc.py \
    outputs/finetune_ewc/whisper_tiny/combined/ami_100h/wer_wrt_steps_on_fab.csv \
    outputs/finetuning/whisper_tiny/ami_100h/wer_wrt_steps_on_fab.csv \
    --is-relative \
    --figsize 8 4


# ----------------------------------------------------------------------------

python scripts/report_utils/forgetting/concat_csv_wer_wrt_steps.py \
    outputs/finetune_ewc/whisper_tiny/preserve_french/norm_only

python scripts/report_utils/forgetting/plot_forgetting_wer_wrt_steps_on_fab.py \
    outputs/finetune_ewc/whisper_tiny/preserve_french/wer_wrt_steps_on_fab.csv

python scripts/report_utils/forgetting/plot_forgetting_wer_wrt_steps_on_fab.py \
    outputs/finetune_ewc/whisper_tiny/preserve_french/wer_wrt_steps_on_fab.csv \
    --is-relative

python scripts/report_utils/forgetting/plot_forgetting_wer_wrt_steps_on_fab-ewc.py \
    outputs/finetune_ewc/whisper_tiny/preserve_french/wer_wrt_steps_on_fab.csv \
    outputs/finetuning/whisper_tiny/ami_100h/wer_wrt_steps_on_fab.csv

python scripts/report_utils/forgetting/plot_forgetting_wer_wrt_steps_on_fab-ewc.py \
    outputs/finetune_ewc/whisper_tiny/preserve_french/wer_wrt_steps_on_fab.csv \
    outputs/finetuning/whisper_tiny/ami_100h/wer_wrt_steps_on_fab.csv \
    --is-relative \
    --figsize 8 4
