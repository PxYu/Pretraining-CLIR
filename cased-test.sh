#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:10
#SBATCH --output=logs/slurm/%j.stdout
#SBATCH --error=logs/slurm/%j.stderr
#SBATCH --job-name=ft-cased
#SBATCH --open-mode=append
#SBATCH --partition=2080Ti

source /mnt/home/puxuan/miniconda3/etc/profile.d/conda.sh

conda activate rtx

./finetune.sh xlmr xlm-roberta-base mix en de $1
./finetune.sh mbert bert-base-multilingual-cased mix en de $1

./finetune.sh xlmr xlm-roberta-base mix en fr $1
./finetune.sh mbert bert-base-multilingual-cased mix en fr $1

./finetune.sh xlmr xlm-roberta-base mix en es $1
./finetune.sh mbert bert-base-multilingual-cased mix en es $1
