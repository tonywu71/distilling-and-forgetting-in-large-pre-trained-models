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
#SBATCH -J distil_k_best-librispeech_clean_100h-hpt
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


echo -e "JobID: $SLURM_JOB_ID\n======" > $LOG
echo "Time: `date`" >> $LOG
echo "Running on master node: `hostname`" >> $LOG
echo "python `which python`": >> $LOG


#! ###########################################################
#! ####                    MAIN                    ###########
#! ###########################################################

# python scripts/distil_whisper.py \
#     configs/distil_configs/k_best/librispeech/seq_level_ranked/hpt/k_3-ranked-hpt-beta_1.yaml \
#     >> $LOG 2> $ERR

# python scripts/distil_whisper.py \
#     configs/distil_configs/k_best/librispeech/seq_level_ranked/hpt/k_3-ranked-hpt-beta_2.yaml \
#     >> $LOG 2> $ERR

python scripts/distil_whisper.py \
    configs/distil_configs/k_best/librispeech/seq_level_ranked/hpt/k_3-ranked-hpt-beta_5.yaml \
    >> $LOG 2> $ERR

# python scripts/distil_whisper.py \
#     configs/distil_configs/k_best/librispeech/seq_level_ranked/hpt/k_3-ranked-hpt-beta_100.yaml \
#     >> $LOG 2> $ERR

#! #############################################


echo "Time: `date`" >> $LOG
