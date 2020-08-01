#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:8
#SBATCH --output=slogs/%j.stdout
#SBATCH --error=slogs/%j.stderr
#SBATCH --job-name=ft-cased
#SBATCH --open-mode=append
#SBATCH --partition=DGX

./finetune.sh xlmr xlm-roberta-base mix en de $1
./finetune.sh mbert bert-base-multilingual-cased mix en de $1

./finetune.sh xlmr xlm-roberta-base mix en fr $1
./finetune.sh mbert bert-base-multilingual-cased mix en fr $1

./finetune.sh xlmr xlm-roberta-base mix en es $1
./finetune.sh mbert bert-base-multilingual-cased mix en es $1
