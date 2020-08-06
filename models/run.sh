#!/bin/bash
#
#SBATCH --job-name=exp{job_id}
#SBATCH --partition=2080ti-long
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=24GB
#SBATCH -d singleton
#SBATCH --mail-type=ALL
#SBATCH --mail-user={username}@cs.umass.edu

# High level experimental details :- {top_details}
# Low leve experimental details :- {lower_details}
mkdir /home/nazaninjafar/ds4cg2020/UMassDS/DS4CG2020-aucode/expjob-name
BASE_PATH="/home/nazaninjafar/ds4cg2020/"
PROJECT_PATH="$BASE_PATH/UMassDS/DS4CG2020-aucode/"
EXPERIMENT_PATH="$PROJECT_PATH/experiments/exp{job_id}"
export TMPDIR=/mnt/nfs/scraich1/{username}/tmp


BASE_PARAMS=( \
  --model bert \
  --epoch 5 \
  )


python main.py \
  "${BASE_PARAMS[@]}"\
 --learningrate 5e-5 >  /home/nazaninjafar/ds4cg2020/UMassDS/DS4CG2020-aucode/expjob-name/log.txt

