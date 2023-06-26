python scripts/report_utils/forgetting/plot_tac_forgetting.py \
    outputs/report/forgetting_wrt_training_steps/tac/tac_finetune_on_librispeech-forgetting_wrt_training_steps-fab.csv \
    --figsize 4 4

python scripts/report_utils/forgetting/plot_tac_forgetting.py \
    outputs/report/forgetting_wrt_training_steps/tac/tac_finetune_on_librispeech-forgetting_wrt_training_steps-fab.csv \
    --is-relative \
    --figsize 4 4


python scripts/report_utils/forgetting/plot_tac_forgetting.py \
    outputs/report/forgetting_wrt_training_steps/tac/ami_10h/tac_finetune_on_ami_10h-forgetting_wrt_training_steps-fab.csv \
    --figsize 4 4

python scripts/report_utils/forgetting/plot_tac_forgetting.py \
    outputs/report/forgetting_wrt_training_steps/tac/ami_10h/tac_finetune_on_ami_10h-forgetting_wrt_training_steps-fab.csv \
    --is-relative \
    --figsize 4 4
