# --- BASE ---
python scripts/report_utils/wer_wrt_model_size/plot_wer_wrt_model_size.py \
    outputs/report/wer_wrt_model_size/wer_wrt_model_size.csv \
    --savename wer_wrt_model_size_initial

python scripts/report_utils/wer_wrt_model_size/plot_wer_wrt_model_size.py \
    outputs/report/wer_wrt_model_size/wer_wrt_model_size.csv \
    --log \
    --savename wer_wrt_model_size_initial-log

python scripts/report_utils/wer_wrt_model_size/plot_wer_wrt_model_size.py \
    outputs/report/wer_wrt_model_size/wer_wrt_model_size.csv \
    --regression \
    --log \
    --savename wer_wrt_model_size_initial-regression-log



# --- IDEAL ---
python scripts/report_utils/wer_wrt_model_size/plot_wer_wrt_model_size.py \
    outputs/report/wer_wrt_model_size/wer_wrt_model_size-ideal.csv \
    --savename wer_wrt_model_size_initial-ideal
