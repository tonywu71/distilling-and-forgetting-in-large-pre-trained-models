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
#SBATCH -J eval_whisper_on_ami_test-finetuned
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
#SBATCH --time=00:20:00
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

python scripts/eval_whisper.py \
    checkpoints/distil_word_level/whisper_medium_to_tiny_unsupervised/hpt/alpha_95e-2_temp_1/final \
    --dataset-name ami \
    --batch-size 1024 \
    >> $LOG 2> $ERR

# python scripts/eval_whisper.py \
#     checkpoints/finetuning/whisper_medium/ami_100h_hpt_reference_teacher/checkpoint-3000 \
#     --dataset-name ami \
#     --batch-size 32 \
#     --no-repeat-ngram-size 6 \
#     --savepath "outputs/finetuning/whisper_medium/ami_100h_hpt_reference_teacher/checkpoint-3000-ami-no_repeat_ngram_6" \
#     >> $LOG 2> $ERR

#! #############################################


echo "Time: `date`" >> $LOG
