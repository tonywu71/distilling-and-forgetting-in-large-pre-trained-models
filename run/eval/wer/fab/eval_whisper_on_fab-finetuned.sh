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
#SBATCH -J eval_whisper_on_fab-finetuned
#! Which project should be charged (NB Wilkes2 projects end in '-GPU'):
#SBATCH -A MLMI-tw581-SL2-GPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total?
#! Note probably this should not exceed the total number of GPUs in use.
#SBATCH --ntasks=1
#! Specify the number of GPUs per node (between 1 and 4; must be 4 if nodes>1).
#! Note that the job submission script will enforce no more than 32 cpus per GPU.
#SBATCH --gres=gpu:1
#! How much wallclock time will be required?
#SBATCH --time=01:00:00
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

# =====================       Vanilla fine-tuning       =====================

# python scripts/eval_whisper.py \
#     checkpoints/finetuning/whisper_tiny/ami_25h_hpt_reference/final \
#     --dataset-name fab \
#     --subset ami --subset librispeech_fr \
#     --batch-size 1024 >> $LOG 2> $ERR

# python scripts/eval_whisper.py \
#     checkpoints/finetuning/whisper_medium/ami_25h_hpt_reference_teacher/final \
#     --dataset-name fab \
#     --subset ami --subset librispeech_fr \
#     --batch-size 128 >> $LOG 2> $ERR


# =====================       1-best       =====================

# python scripts/eval_whisper.py \
#     checkpoints/distil_1_best/whisper_medium_to_tiny/ami_100h-3_epochs/final \
#     --dataset-name fab \
#     --subset ami --subset librispeech_fr \
#     --batch-size 1024 >> $LOG 2> $ERR

python scripts/eval_whisper.py \
    checkpoints/distil_word_level/whisper_medium_to_tiny/librispeech/alpha_5e-1_temp_1-no_freeze/final \
    --dataset-name fab \
    --subset librispeech_en_clean \
    --batch-size 1024 >> $LOG 2> $ERR

# =====================       Word-level       =====================

# python scripts/eval_whisper.py \
#     checkpoints/distil_word_level/whisper_finetuned_medium_to_pre_finetuned_tiny/hpt/alpha_5e-1_temp_1/final \
#     --dataset-name fab \
#     --subset ami --subset librispeech_fr \
#     --batch-size 1024 >> $LOG 2> $ERR

# =====================       EWC       =====================

# python scripts/eval_whisper.py \
#     checkpoints/finetune_ewc/whisper_tiny/ami_100h-lambda_1e-4/final \
#     --dataset-name fab \
#     --subset ami --subset librispeech_fr \
#     --batch-size 1024 >> $LOG 2> $ERR


# =====================       TAC       =====================

# python scripts/eval_whisper.py \
#     checkpoints/finetune_tac/whisper_tiny/hpt/gamma_1e+0/final \
#     --dataset-name fab \
#     --subset ami --subset librispeech_fr \
#     --batch-size 1024 >> $LOG 2> $ERR

#! #############################################


echo "Time: `date`" >> $LOG
