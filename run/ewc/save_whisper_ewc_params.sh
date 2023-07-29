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
#SBATCH -J save_whisper_ewc_params
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
#SBATCH --time=07:00:00
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


echo -e "JobID: $SLURM_JOB_ID\n======" > $LOG
echo "Time: `date`" >> $LOG
echo "Running on master node: `hostname`" >> $LOG
echo "python `which python`": >> $LOG


#! ###########################################################
#! ####                    MAIN                    ###########
#! ###########################################################

python scripts/save_whisper_ewc_params.py \
    openai/whisper-tiny \
    --language french \
    --task transcribe \
    --dataset-name mls_french_diagnostic \
    --split train \
    >> $LOG 2> $ERR

# python scripts/save_whisper_ewc_params.py openai/whisper-tiny \
#     --language english \
#     --task transcribe \
#     --dataset-name ami_100h \
#     --skip-lowercase \
#     --split train \
#     --cache-dir /home/tw581/rds/rds-altaslp-8YSp2LXTlkY/experiments/tw581/cache/huggingface/preprocessed_datasets/ami_100h/multilingual_tokenizer/train >> $LOG 2> $ERR

#! #############################################


echo "Time: `date`" >> $LOG
