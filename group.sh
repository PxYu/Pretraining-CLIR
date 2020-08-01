#!/bin/bash
# here the script
# 1. runs with bash (because zsh is not available on cluster)
# 2. test multiple pre-trained models at the same time
# 3. test all language pairs at the same time

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:8
#SBATCH --output=slogs/%j.stdout
#SBATCH --error=slogs/%j.stderr
#SBATCH --job-name=ft-short
#SBATCH --open-mode=append
#SBATCH --partition=DGX

pretrained_dirs=(
        "bert-base-multilingual-uncased"
        "dumped/short_qlm-rr/2147416/huggingface-4"
	"dumped/short_qlm-rr/2147416/huggingface-9"
        #"/mnt/home/puxuan/XLM-Trainer/mBertLong/mBert-base-p1024-w64-0"
        #"dumped/long_qlm-rr/2148591/huggingface-0"
	#"dumped/long_qlm-rr/2150176/huggingface-1"
	)

for modelpath in ${pretrained_dirs[@]}
do
    sh finetune.sh $1 $modelpath $2 $3 $4 $5
done
