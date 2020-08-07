#!/bin/bash
# here the script
# 1. runs with bash (because zsh is not available on cluster)
# 2. test multiple pre-trained models at the same time
# 3. test all language pairs at the same time

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:10
#SBATCH --output=logs/slurm/%j.stdout
#SBATCH --error=logs/slurm/%j.stderr
#SBATCH --job-name=ft-long
#SBATCH --open-mode=append
#SBATCH --partition=2080Ti

source /mnt/home/puxuan/miniconda3/etc/profile.d/conda.sh

conda activate rtx

pretrained_dirs=(
        # "bert-base-multilingual-uncased"
        # "/mnt/scratch/puxuan/short_qlm-rr/2147416/huggingface-4"
	# "/mnt/scratch/puxuan/short_qlm-rr/2147416/huggingface-9"
        "/mnt/home/puxuan/XLM-Trainer/mBertLong/mBert-base-p1024-w64-0"
        "/mnt/scratch/puxuan/long_qlm-rr/2148591/huggingface-0"
	"/mnt/scratch/puxuan/long_qlm-rr/2150176/huggingface-1"
	# "/mnt/scratch/puxuan/long_qlm-rr/2151242/huggingface-0"
	)

for modelpath in ${pretrained_dirs[@]}
do
    echo $1 $modelpath $2 $3 $4 $5
    ./uncased-finetune.sh $1 $modelpath $2 $3 $4 $5
done
