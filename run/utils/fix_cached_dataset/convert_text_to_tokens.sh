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
#SBATCH -J convert_text_to_tokens
#! Which project should be charged (NB Wilkes2 projects end in '-GPU'):
#SBATCH -A MLMI-tw581-SL2-CPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How much wallclock time will be required?
#SBATCH --time=05:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=NONE
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue

#! Do not change (CPU-only partition):
#SBATCH -p cclake-himem
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

python scripts/utils/convert_text_to_tokens.py \
    /home/tw581/rds/rds-altaslp-8YSp2LXTlkY/experiments/tw581/cache/huggingface/k_beam_search_cache/ami_100h/checkpoints/finetuning/whisper_medium/ami_100h_hpt_reference_teacher/checkpoint-3000/k_1_old \
    /home/tw581/rds/rds-altaslp-8YSp2LXTlkY/experiments/tw581/cache/huggingface/k_beam_search_cache/ami_100h/checkpoints/finetuning/whisper_medium/ami_100h_hpt_reference_teacher/checkpoint-3000/k_1 >> $LOG 2> $ERR

#! #############################################


echo "Time: `date`" >> $LOG
