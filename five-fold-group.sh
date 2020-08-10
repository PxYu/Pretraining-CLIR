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
#SBATCH --job-name=ft-5f
#SBATCH --open-mode=append
#SBATCH --partition=2080Ti

source /mnt/home/puxuan/miniconda3/etc/profile.d/conda.sh

conda activate rtx

pretrained_dirs=(
        # "bert-base-multilingual-uncased"
        # "/mnt/scratch/puxuan/short_qlm-rr/2147416/huggingface-4"
	    # "/mnt/scratch/puxuan/short_qlm-rr/2147416/huggingface-9"
        # "/mnt/home/puxuan/XLM-Trainer/mBertLong/mBert-base-p1024-w64-0"
        # "/mnt/scratch/puxuan/long_qlm-rr/2148591/huggingface-0"
        # "/mnt/scratch/puxuan/long_qlm-rr/2150176/huggingface-1"
        # "/mnt/scratch/puxuan/long_qlm-rr/2151242/huggingface-0"
        # "xlm-roberta-base"
        # "/mnt/scratch/puxuan/short_rr/2186734/huggingface-4"
        # "/mnt/scratch/puxuan/short_rr/2186734/huggingface-9"
        # "/mnt/scratch/puxuan/short_rr/2186734/huggingface-14"
        # "/mnt/scratch/puxuan/short_rr/2186734/huggingface-19"
        # "/mnt/scratch/puxuan/short_qlm/2186735/huggingface-4"
        # "/mnt/scratch/puxuan/short_qlm/2186735/huggingface-9"
        # "/mnt/scratch/puxuan/short_qlm/2186735/huggingface-14"
        # "/mnt/scratch/puxuan/short_qlm/2186735/huggingface-19"
        # "/mnt/scratch/puxuan/short_qlm-rr/2186738/huggingface-4"
        # "/mnt/scratch/puxuan/short_qlm-rr/2186738/huggingface-9"
        # "/mnt/scratch/puxuan/long_qlm-rr/2186740/huggingface-2"
        # "/mnt/scratch/puxuan/long_qlm-rr/2186742/huggingface-3"
        # "/mnt/scratch/puxuan/short_qlm_0.15/2186792/huggingface-4"
        # "/mnt/scratch/puxuan/short_qlm_0.15/2186792/huggingface-9"
        # "/mnt/scratch/puxuan/short_qlm_0.15/2186792/huggingface-14"
        "/mnt/scratch/puxuan/short_qlm_0.15/2186792/huggingface-19"
        "/mnt/scratch/puxuan/short_qlm_0.45/2186793/huggingface-4"
        # "/mnt/scratch/puxuan/short_qlm_0.45/2186793/huggingface-9"
        # "/mnt/scratch/puxuan/short_qlm_0.45/2186793/huggingface-14"
        "/mnt/scratch/puxuan/short_qlm_0.45/2186793/huggingface-19"
        # "/mnt/scratch/puxuan/short_qlm_doc/2186796/huggingface-4"
        # "/mnt/scratch/puxuan/short_qlm_doc/2186796/huggingface-9"
        # "/mnt/scratch/puxuan/short_qlm_doc/2186796/huggingface-14"
        # "/mnt/scratch/puxuan/short_qlm_doc/2186796/huggingface-19"
        "/mnt/scratch/puxuan/short_qlm_mix/2192279/huggingface-4"
        "/mnt/scratch/puxuan/short_qlm_mix/2192279/huggingface-9"
        "/mnt/scratch/puxuan/short_qlm_mix/2192279/huggingface-14"
        "/mnt/scratch/puxuan/short_qlm_mix/2192279/huggingface-19"

	)

for modelpath in ${pretrained_dirs[@]}
do
    echo $1 $modelpath $2 $3 $4 $5
    ./five-fold-ft.sh $1 $modelpath $2 $3 $4 $5
done
