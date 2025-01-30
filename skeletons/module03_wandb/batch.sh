#!/bin/bash

# Andrew H. Fagg
#
# Example with one experiment
#
# When you use this batch file:
#  Change the email address to yours! (I don't want email about your experiments!)
#  Change the chdir line to match the location of where your code is located
#
# Reasonable partitions: debug_5min, debug_30min, normal, debug_gpu, gpu
#

#
#SBATCH --partition=debug_5min
#SBATCH --ntasks=1
#SBATCH --mem=1G

# The %j is translated into the job number
#SBATCH --output=results/xor_%j_stdout.txt
#SBATCH --error=results/xor_%j_stderr.txt

#SBATCH --time=00:05:00
#SBATCH --job-name=xor_test
#SBATCH --mail-user=andrewhfagg@gmail.com #INSERT_YOUR_EMAIL_ADDRESS_HERE
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/fagg/deep_learning_practice/skeletons/module03_wandb
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up

. /home/fagg/tf_setup.sh
conda activate dnn

# Change this line to start an instance of your experiment
python xor_base_solution.py --exp 0 --epochs 10000 --render --label xortest -vvv --exp 1



