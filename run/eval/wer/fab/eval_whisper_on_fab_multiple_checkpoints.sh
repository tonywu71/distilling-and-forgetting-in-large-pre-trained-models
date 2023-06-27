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
#SBATCH -J eval_whisper_on_fab_multiple_checkpoints
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
#SBATCH --time=02:00:00
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

# python scripts/eval_whisper_multiple_checkpoints.py \
#     checkpoints/finetuning/whisper_tiny/librispeech_clean_100h-benchmark/checkpoint-500 \
#     checkpoints/finetuning/whisper_tiny/librispeech_clean_100h-benchmark/checkpoint-1000 \
#     checkpoints/finetuning/whisper_tiny/librispeech_clean_100h-benchmark/checkpoint-1500 \
#     checkpoints/finetuning/whisper_tiny/librispeech_clean_100h-benchmark/checkpoint-2000 \
#     checkpoints/finetuning/whisper_tiny/librispeech_clean_100h-benchmark/checkpoint-2500 \
#     checkpoints/finetune_tac/whisper_tiny/librispeech_clean_100h-gamma_1e-2/checkpoint-500 \
#     checkpoints/finetune_tac/whisper_tiny/librispeech_clean_100h-gamma_1e-2/checkpoint-1000 \
#     checkpoints/finetune_tac/whisper_tiny/librispeech_clean_100h-gamma_1e-2/checkpoint-1500 \
#     checkpoints/finetune_tac/whisper_tiny/librispeech_clean_100h-gamma_1e-2/checkpoint-2000 \
#     checkpoints/finetune_tac/whisper_tiny/librispeech_clean_100h-gamma_1e-2/checkpoint-2500 \
#     checkpoints/finetune_tac/whisper_tiny/librispeech_clean_100h-gamma_2e-1/checkpoint-500 \
#     checkpoints/finetune_tac/whisper_tiny/librispeech_clean_100h-gamma_2e-1/checkpoint-1000 \
#     checkpoints/finetune_tac/whisper_tiny/librispeech_clean_100h-gamma_2e-1/checkpoint-1500 \
#     checkpoints/finetune_tac/whisper_tiny/librispeech_clean_100h-gamma_2e-1/checkpoint-2000 \
#     checkpoints/finetune_tac/whisper_tiny/librispeech_clean_100h-gamma_2e-1/checkpoint-2500 \
#     --dataset-name fab \
#     --subset librispeech_en_clean --subset librispeech_fr --subset librispeech_pt \
#     --batch-size 256 \
#     >> $LOG 2> $ERR

# python scripts/eval_whisper_multiple_checkpoints.py \
#     checkpoints/finetuning/whisper_tiny/ami_10h-benchmark/checkpoint-68 \
#     checkpoints/finetuning/whisper_tiny/ami_10h-benchmark/checkpoint-136 \
#     checkpoints/finetuning/whisper_tiny/ami_10h-benchmark/checkpoint-204 \
#     checkpoints/finetuning/whisper_tiny/ami_10h-benchmark/checkpoint-272 \
#     checkpoints/finetuning/whisper_tiny/ami_10h-benchmark/checkpoint-340 \
#     checkpoints/finetune_tac/whisper_tiny/ami_10h-gamma_1e-2/checkpoint-68 \
#     checkpoints/finetune_tac/whisper_tiny/ami_10h-gamma_1e-2/checkpoint-136 \
#     checkpoints/finetune_tac/whisper_tiny/ami_10h-gamma_1e-2/checkpoint-204 \
#     checkpoints/finetune_tac/whisper_tiny/ami_10h-gamma_1e-2/checkpoint-272 \
#     checkpoints/finetune_tac/whisper_tiny/ami_10h-gamma_1e-2/checkpoint-340 \
#     checkpoints/finetune_tac/whisper_tiny/ami_10h-gamma_1e-1/checkpoint-68 \
#     checkpoints/finetune_tac/whisper_tiny/ami_10h-gamma_1e-1/checkpoint-136 \
#     checkpoints/finetune_tac/whisper_tiny/ami_10h-gamma_1e-1/checkpoint-204 \
#     checkpoints/finetune_tac/whisper_tiny/ami_10h-gamma_1e-1/checkpoint-272 \
#     checkpoints/finetune_tac/whisper_tiny/ami_10h-gamma_1e-1/checkpoint-340 \
#     --dataset-name fab \
#     --subset librispeech_en_clean --subset librispeech_fr --subset librispeech_pt \
#     --batch-size 256 \
#     >> $LOG 2> $ERR

python scripts/eval_whisper_multiple_checkpoints.py \
    checkpoints/finetuning/whisper_tiny/ami_10h-benchmark/checkpoint-204 \
    checkpoints/finetune_tac/whisper_tiny/ami_10h-gamma_1e-2/checkpoint-204 \
    checkpoints/finetune_tac/whisper_tiny/ami_10h-gamma_1e-1/checkpoint-204 \
    --dataset-name fab_with_ami_10h \
    --subset ami_10h --subset librispeech_fr --subset librispeech_pt \
    --batch-size 256 \
    >> $LOG 2> $ERR


#! #############################################


echo "Time: `date`" >> $LOG
