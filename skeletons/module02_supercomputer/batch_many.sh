#!/bin/bash
# Andrew H. Fagg
#
# Example with an array of experiments
#  The --array line says that we will execute 4 experiments (numbered 0,1,2,3).
#   You can specify ranges or comma-separated lists on this line
#  For each experiment, the SLURM_ARRAY_TASK_ID will be set to the experiment number
#   In this case, this ID is used to set the name of the stdout/stderr file names
#   and is passed as an argument to the python program
#
#
# When you use this batch file:
#  Change the email address to yours! (I don't want email about your experiments)
#  Change the chdir line to match the location of where your code is located
#
# Reasonable partitions: debug_5min, debug_30min, normal
#

#SBATCH --partition=debug_5min
#SBATCH --ntasks=1
#SBATCH --mem=1G
# The %j is translated into the job number
#SBATCH --output=logs/xor_%j_stdout.txt
#SBATCH --error=logs/xor_%j_stderr.txt
#SBATCH --time=00:05:00
#SBATCH --job-name=xor_test
#SBATCH --mail-user=YOUR EMAIL HERE
#SBATCH --mail-type=ALL
#SBATCH --chdir=INSERT_YOUR_EXPERIMENT_DIRECTORY_HERE
#SBATCH --array=0-3
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up

. /home/fagg/tf_setup.sh
conda activate dnn

# Change this line to configure an instance of your experiment
python xor_base.py --epochs 10 --exp $SLURM_ARRAY_TASK_ID
