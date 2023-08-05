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
#SBATCH -J eval_whisper_on_ami_validation_multiple_checkpoints
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

python scripts/eval_whisper_multiple_checkpoints.py \
    checkpoints/distil_k_best/whisper_medium_to_tiny/ami_100h/k_3/ranked/hpt/beta_1/checkpoint-3000 \
    checkpoints/distil_k_best/whisper_medium_to_tiny/ami_100h/k_3/ranked/hpt/beta_2/checkpoint-3000 \
    checkpoints/distil_k_best/whisper_medium_to_tiny/ami_100h/k_3/ranked/hpt/beta_5/checkpoint-3000 \
    checkpoints/distil_k_best/whisper_medium_to_tiny/ami_100h/k_3/ranked/hpt/beta_10/checkpoint-3000 \
    checkpoints/distil_k_best/whisper_medium_to_tiny/ami_100h/k_3/uniform/partial/checkpoint-3000 \
    --dataset-name ami_validation \
    --batch-size 1024 \
    >> $LOG 2> $ERR

#! #############################################


echo "Time: `date`" >> $LOG
