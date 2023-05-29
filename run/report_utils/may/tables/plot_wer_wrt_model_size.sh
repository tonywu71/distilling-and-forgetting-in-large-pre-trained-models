# --- BASE ---
python scripts/report_utils/wer_wrt_model_size/plot_wer_wrt_model_size.py \
    outputs/report/wer_wrt_model_size/wer_wrt_model_size-ami.csv \
    --plot-ideal \
    --savename wer_wrt_model_size-ami

# python scripts/report_utils/wer_wrt_model_size/plot_wer_wrt_model_size.py \
#     outputs/report/wer_wrt_model_size/wer_wrt_model_size-ami.csv \
#     --log \
#     --savename wer_wrt_model_size-ami-log

# python scripts/report_utils/wer_wrt_model_size/plot_wer_wrt_model_size.py \
#     outputs/report/wer_wrt_model_size/wer_wrt_model_size-ami.csv \
#     --regression \
#     --log \
#     --savename wer_wrt_model_size-ami-log-regression
