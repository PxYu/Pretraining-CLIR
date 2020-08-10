#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm/%j.stdout
#SBATCH --error=logs/slurm/%j.stderr
#SBATCH --job-name=ft-qa
#SBATCH --open-mode=append
#SBATCH --partition=2080Ti

source /mnt/home/puxuan/miniconda3/etc/profile.d/conda.sh
conda activate rtx

export TOKENIZERS_PARALLELISM=false

# srun python finetune-qa.py --model_type mbert --model_name /mnt/scratch/puxuan/short_qlm-rr/2186738/huggingface-4 --batch_size 32 --num_epochs 10
# srun python finetune-qa.py --model_type mbert --model_name /mnt/scratch/puxuan/short_qlm-rr/2186738/huggingface-9 --batch_size 32 --num_epochs 10


# srun python finetune-qa.py --model_type mbert --model_name /mnt/scratch/puxuan/short_qlm/2186735/huggingface-4 --batch_size 32 --num_epochs 10
# srun python finetune-qa.py --model_type mbert --model_name /mnt/scratch/puxuan/short_qlm/2186735/huggingface-9 --batch_size 32 --num_epochs 10
# srun python finetune-qa.py --model_type mbert --model_name /mnt/scratch/puxuan/short_qlm/2186735/huggingface-14 --batch_size 32 --num_epochs 10
# srun python finetune-qa.py --model_type mbert --model_name /mnt/scratch/puxuan/short_qlm/2186735/huggingface-19 --batch_size 32 --num_epochs 10

# srun python finetune-qa.py --model_type mbert --model_name /mnt/scratch/puxuan/short_rr/2186734/huggingface-4 --batch_size 32 --num_epochs 10
# srun python finetune-qa.py --model_type mbert --model_name /mnt/scratch/puxuan/short_rr/2186734/huggingface-9 --batch_size 32 --num_epochs 10
# srun python finetune-qa.py --model_type mbert --model_name /mnt/scratch/puxuan/short_rr/2186734/huggingface-14 --batch_size 32 --num_epochs 10
# srun python finetune-qa.py --model_type mbert --model_name /mnt/scratch/puxuan/short_rr/2186734/huggingface-19 --batch_size 32 --num_epochs 10

# srun python finetune-qa.py --model_type mbert-long --model_name /mnt/scratch/puxuan/long_qlm-rr/2186740/huggingface-2 --batch_size 32 --num_epochs 10
srun python finetune-qa.py --model_type mbert-long --model_name /mnt/scratch/puxuan/long_qlm-rr/2186742/huggingface-3 --batch_size 32 --num_epochs 10

