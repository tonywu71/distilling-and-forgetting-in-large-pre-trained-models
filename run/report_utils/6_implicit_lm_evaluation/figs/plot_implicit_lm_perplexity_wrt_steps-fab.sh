# ====================================================================================================
# Generate CSV from eval outputs:
python scripts/report_utils/implicit_lm/concat_csv_forgetting_perplexity.py \
    outputs/finetuning/whisper_tiny/ami_100h/ppl_wrt_steps_on_fad/checkpoint-0-implicit_lm-ppl-fad.csv \
    outputs/finetuning/whisper_tiny/ami_100h/ppl_wrt_steps_on_fad/checkpoint-600-implicit_lm-ppl-fad.csv \
    outputs/finetuning/whisper_tiny/ami_100h/ppl_wrt_steps_on_fad/checkpoint-1200-implicit_lm-ppl-fad.csv \
    outputs/finetuning/whisper_tiny/ami_100h/ppl_wrt_steps_on_fad/checkpoint-1800-implicit_lm-ppl-fad.csv \
    outputs/finetuning/whisper_tiny/ami_100h/ppl_wrt_steps_on_fad/checkpoint-2400-implicit_lm-ppl-fad.csv \
    outputs/finetuning/whisper_tiny/ami_100h/ppl_wrt_steps_on_fad/checkpoint-3000-implicit_lm-ppl-fad.csv

# Generate plot from previous CSV:
python scripts/report_utils/implicit_lm/plot_forgetting_perplexity_wrt_checkpoints.py \
    outputs/finetuning/whisper_tiny/ami_100h/ppl_wrt_steps_on_fad/concat_csv_forgetting_perplexity_on_fad.csv

python scripts/report_utils/implicit_lm/plot_forgetting_perplexity_wrt_checkpoints.py \
    outputs/finetuning/whisper_tiny/ami_100h/ppl_wrt_steps_on_fad/concat_csv_forgetting_perplexity_on_fad.csv \
    --is-relative


# ====================================================================================================
# Generate CSV from eval outputs:
python scripts/report_utils/implicit_lm/concat_csv_forgetting_perplexity.py \
    outputs/finetuning/whisper_tiny/ami_100h/ppl_wrt_steps_on_mlsdc/checkpoint-0-implicit_lm-ppl-mlsdc.csv \
    outputs/finetuning/whisper_tiny/ami_100h/ppl_wrt_steps_on_mlsdc/checkpoint-600-implicit_lm-ppl-mlsdc.csv \
    outputs/finetuning/whisper_tiny/ami_100h/ppl_wrt_steps_on_mlsdc/checkpoint-1200-implicit_lm-ppl-mlsdc.csv \
    outputs/finetuning/whisper_tiny/ami_100h/ppl_wrt_steps_on_mlsdc/checkpoint-1800-implicit_lm-ppl-mlsdc.csv \
    outputs/finetuning/whisper_tiny/ami_100h/ppl_wrt_steps_on_mlsdc/checkpoint-2400-implicit_lm-ppl-mlsdc.csv \
    outputs/finetuning/whisper_tiny/ami_100h/ppl_wrt_steps_on_mlsdc/checkpoint-3000-implicit_lm-ppl-mlsdc.csv

# Generate plot from previous CSV:
python scripts/report_utils/implicit_lm/plot_forgetting_perplexity_wrt_checkpoints.py \
    outputs/finetuning/whisper_tiny/ami_100h/ppl_wrt_steps_on_mlsdc/concat_csv_forgetting_perplexity_on_mlsdc.csv

python scripts/report_utils/implicit_lm/plot_forgetting_perplexity_wrt_checkpoints.py \
    outputs/finetuning/whisper_tiny/ami_100h/ppl_wrt_steps_on_mlsdc/concat_csv_forgetting_perplexity_on_mlsdc.csv \
    --is-relative


# ====================================================================================================
# Generate CSV from eval outputs:
python scripts/report_utils/implicit_lm/concat_csv_forgetting_perplexity.py \
    outputs/finetuning/whisper_tiny/ami_100h/ppl_wrt_steps_custom/checkpoint-0-implicit_lm-ppl-custom.csv \
    outputs/finetuning/whisper_tiny/ami_100h/ppl_wrt_steps_custom/checkpoint-600-implicit_lm-ppl-custom.csv \
    outputs/finetuning/whisper_tiny/ami_100h/ppl_wrt_steps_custom/checkpoint-1200-implicit_lm-ppl-custom.csv \
    outputs/finetuning/whisper_tiny/ami_100h/ppl_wrt_steps_custom/checkpoint-1800-implicit_lm-ppl-custom.csv \
    outputs/finetuning/whisper_tiny/ami_100h/ppl_wrt_steps_custom/checkpoint-2400-implicit_lm-ppl-custom.csv \
    outputs/finetuning/whisper_tiny/ami_100h/ppl_wrt_steps_custom/checkpoint-3000-implicit_lm-ppl-custom.csv

# Generate plot from previous CSV:
python scripts/report_utils/implicit_lm/plot_forgetting_perplexity_wrt_checkpoints.py \
    outputs/finetuning/whisper_tiny/ami_100h/ppl_wrt_steps_custom/concat_csv_forgetting_perplexity.csv \
    --legend-title "Language"

python scripts/report_utils/implicit_lm/plot_forgetting_perplexity_wrt_checkpoints.py \
    outputs/finetuning/whisper_tiny/ami_100h/ppl_wrt_steps_custom/concat_csv_forgetting_perplexity.csv \
    --legend-title "Language" \
    --is-relative
