#!/bin/bash
#!
#! Example SLURM job script for Wilkes3 (AMD EPYC 7763, ConnectX-6, A100)
#! Last updated: Fri 30 Jul 11:07:58 BST 2021
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################
#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J eval_whisper_on_ami_test-vanilla-compare_decoding
#! Which project should be charged (NB Wilkes2 projects end in '-GPU'):
#SBATCH -A DUDLEY-SL3-GPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total?
#! Note probably this should not exceed the total number of GPUs in use.
#SBATCH --ntasks=1
#! Specify the number of GPUs per node (between 1 and 4; must be 4 if nodes>1).
#! Note that the job submission script will enforce no more than 32 cpus per GPU.
#SBATCH --gres=gpu:1
#! How much wallclock time will be required?
#SBATCH --time=01:20:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=NONE
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue

#! Do not change:
#SBATCH -p ampere
#! ############################################################


LOGDIR=logs/
DIRPATH_EXP=logs/$SLURM_JOB_NAME/
mkdir -p $DIRPATH_EXP

LOG=$DIRPATH_EXP/$SLURM_JOB_ID.log
ERR=$DIRPATH_EXP/$SLURM_JOB_ID.err


echo -e "JobID: $JOBID\n======" > $LOG
echo "Time: `date`" >> $LOG
echo "Running on master node: `hostname`" >> $LOG
echo "python `which python`": >> $LOG


#! ###########################################################
#! ####                    MAIN                    ###########
#! ###########################################################

# python scripts/eval_whisper.py \
#     openai/whisper-tiny \
#     --dataset-name ami \
#     --batch-size 1024 \
#     --no-repeat-ngram-size 2 \
#     --savepath "./outputs/vanilla/tiny/compare_decoding/no_repeat_ngram_2" \
#     >> $LOG 2> $ERR

# python scripts/eval_whisper.py \
#     openai/whisper-tiny \
#     --dataset-name ami \
#     --batch-size 1024 \
#     --no-repeat-ngram-size 3 \
#     --savepath "./outputs/vanilla/tiny/compare_decoding/no_repeat_ngram_3" \
#     >> $LOG 2> $ERR

# python scripts/eval_whisper.py \
#     openai/whisper-tiny \
#     --dataset-name ami \
#     --batch-size 1024 \
#     --no-repeat-ngram-size 4 \
#     --savepath "./outputs/vanilla/tiny/compare_decoding/no_repeat_ngram_4" \
#     >> $LOG 2> $ERR

# python scripts/eval_whisper.py \
#     openai/whisper-tiny \
#     --dataset-name ami \
#     --batch-size 1024 \
#     --no-repeat-ngram-size 5 \
#     --savepath "./outputs/vanilla/tiny/compare_decoding/no_repeat_ngram_5" \
#     >> $LOG 2> $ERR

# python scripts/eval_whisper.py \
#     openai/whisper-tiny \
#     --dataset-name ami \
#     --batch-size 1024 \
#     --no-repeat-ngram-size 6 \
#     --savepath "./outputs/vanilla/tiny/compare_decoding/no_repeat_ngram_6" \
#     >> $LOG 2> $ERR

# python scripts/eval_whisper.py \
#     openai/whisper-tiny \
#     --dataset-name ami \
#     --batch-size 1024 \
#     --no-repeat-ngram-size 7 \
#     --savepath "./outputs/vanilla/tiny/compare_decoding/no_repeat_ngram_7" \
#     >> $LOG 2> $ERR

# python scripts/eval_whisper.py \
#     openai/whisper-tiny \
#     --dataset-name ami \
#     --batch-size 128 \
#     --num-beams 5 \
#     --savepath "./outputs/vanilla/tiny/compare_decoding/5_beam" \
#     >> $LOG 2> $ERR

python scripts/eval_whisper.py \
    openai/whisper-tiny \
    --dataset-name ami \
    --batch-size 128 \
    --num-beams 5 \
    --no-repeat-ngram-size 6 \
    --savepath "./outputs/vanilla/tiny/compare_decoding/5_beam_no_repeat_ngram_6" \
    >> $LOG 2> $ERR

# python scripts/eval_whisper.py \
#     openai/whisper-tiny \
#     --dataset-name ami \
#     --batch-size 1024 \
#     --sampling \
#     --seed 42 \
#     --savepath "./outputs/vanilla/tiny/compare_decoding/sampling" \
#     >> $LOG 2> $ERR

# python scripts/eval_whisper.py \
#     openai/whisper-tiny \
#     --dataset-name ami \
#     --batch-size 1024 \
#     --sampling \
#     --gen-top-k 40 \
#     --gen-temperature 0.7 \
#     --seed 42 \
#     --savepath "./outputs/vanilla/tiny/compare_decoding/top_k_and_temperature" \
#     >> $LOG 2> $ERR

# python scripts/eval_whisper.py \
#     openai/whisper-tiny \
#     --dataset-name ami \
#     --batch-size 1024 \
#     --sampling \
#     --gen-top-p 0.92 \
#     --seed 42 \
#     --savepath "./outputs/vanilla/tiny/compare_decoding/top_p" \
#     >> $LOG 2> $ERR

# python scripts/eval_whisper.py \
#     openai/whisper-tiny \
#     --dataset-name ami \
#     --batch-size 1024 \
#     --sampling \
#     --gen-top-k 40 \
#     --gen-temperature 0.7 \
#     --gen-top-p 0.92 \
#     --seed 42 \
#     --savepath "./outputs/vanilla/tiny/compare_decoding/top_k_temperature_and_top_p" \
#     >> $LOG 2> $ERR

#! #############################################


echo "Time: `date`" >> $LOG
