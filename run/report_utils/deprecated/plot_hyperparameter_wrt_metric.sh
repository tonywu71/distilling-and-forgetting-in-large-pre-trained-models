python scripts/report_utils/hpt/plot_hyperparameter_wrt_metric.py \
    run/distil/word_level/sequential_hpt/word_level-hpt_results.csv \
    -x "alpha_ce" \
    -y "validation_wer" \
    --r2 "upper right" \
    --filename "hpt_word_error_rate-temperature"
