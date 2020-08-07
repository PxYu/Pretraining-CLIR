#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:8
#SBATCH --output=logs/slurm/%j.stdout
#SBATCH --error=logs/slurm/%j.stderr
#SBATCH --job-name=mbert-l
#SBATCH --open-mode=append
#SBATCH --partition=DGX

LOG_STDOUT="slogs/$SLURM_JOB_ID.stdout"
LOG_STDERR="slogs/$SLURM_JOB_ID.stderr"

# Start (or restart) experiment
date >> $LOG_STDOUT
which python >> $LOG_STDOUT
echo "---Beginning program ---" >> $LOG_STDOUT
echo "Exp name     : test" >> $LOG_STDOUT
echo "SLURM Job ID : $SLURM_JOB_ID" >> $LOG_STDOUT

source /mnt/home/puxuan/miniconda3/etc/profile.d/conda.sh

conda activate dgx

# srun --label python train.py --exp_name short_qlm-rr --fp16 true --amp 2 --model_type mbert --model_path bert-base-multilingual-uncased --batch_size 4 --qlm_mask_mode query --mlm_probability 0.30 --optimizer adam,lr=0.00001 --clip_grad_norm 5 --epoch_size 1000 --max_epoch 20 --accumulate_gradients 1 --lambda_qlm 1 --lambda_rr 1 --num_neg 1 --rr_steps enfr,enes,ende,fres,frde,esde --qlm_steps enfr,enes,ende,fres,frde,esde --max_pairs 100000 --master_port 12223

# srun --label python train.py --exp_name long_qlm-rr --fp16 true --amp 2 --model_type mbert-long --model_path /mnt/scratch/puxuan/long_qlm-rr/2150176/huggingface-3 --batch_size 2 --qlm_mask_mode query --mlm_probability 0.30 --optimizer adam,lr=0.00001 --clip_grad_norm 5 --epoch_size 2000 --max_epoch 4 --accumulate_gradients 1 --lambda_qlm 1 --lambda_rr 1 --num_neg 1 --rr_steps enfr,enes,ende,fres,frde,esde --qlm_steps enfr,enes,ende,fres,frde,esde --max_pairs 100000 --master_port 12223

# srun --label python train.py --exp_name long_qlm-rr --fp16 true --amp 2 --model_type mbert-long --model_path dumped/long_qlm-rr/2186740/huggingface-3 --batch_size 2 --qlm_mask_mode query --mlm_probability 0.30 --optimizer adam,lr=0.00001 --clip_grad_norm 5 --epoch_size 2000 --max_epoch 4 --accumulate_gradients 1 --lambda_qlm 1 --lambda_rr 1 --num_neg 1 --rr_steps enfr,enes,ende,fres,frde,esde --qlm_steps enfr,enes,ende,fres,frde,esde --max_pairs 100000 --master_port 12223

# srun --label python train.py --exp_name short_qlm-rr --fp16 true --amp 2 --model_type mbert --model_path /mnt/scratch/puxuan/short_qlm-rr/2147416/huggingface-9 --batch_size 4 --qlm_mask_mode query --mlm_probability 0.30 --optimizer adam,lr=0.00001 --clip_grad_norm 5 --epoch_size 1000 --max_epoch 10 --accumulate_gradients 1 --lambda_qlm 1 --lambda_rr 1 --num_neg 1 --rr_steps enfr,enes,ende,fres,frde,esde --qlm_steps enfr,enes,ende,fres,frde,esde --max_pairs 100000 --master_port 12223

# srun --label python train.py --exp_name short_rr --fp16 true --amp 2 --model_type mbert --model_path bert-base-multilingual-uncased --batch_size 4 --qlm_mask_mode query --mlm_probability 0.30 --optimizer adam,lr=0.00001 --clip_grad_norm 5 --epoch_size 1000 --max_epoch 20 --accumulate_gradients 1 --lambda_qlm 1 --lambda_rr 1 --num_neg 1 --rr_steps enfr,enes,ende,fres,frde,esde --max_pairs 100000 --master_port 12223

# srun --label python train.py --exp_name short_qlm --fp16 true --amp 2 --model_type mbert --model_path bert-base-multilingual-uncased --batch_size 4 --qlm_mask_mode query --mlm_probability 0.30 --optimizer adam,lr=0.00001 --clip_grad_norm 5 --epoch_size 1000 --max_epoch 20 --accumulate_gradients 1 --lambda_qlm 1 --lambda_rr 1 --num_neg 1 --qlm_steps enfr,enes,ende,fres,frde,esde --max_pairs 100000 --master_port 12223

# srun --label python train.py --exp_name short_qlm_0.15 --fp16 true --amp 2 --model_type mbert --model_path bert-base-multilingual-uncased --batch_size 4 --qlm_mask_mode query --mlm_probability 0.15 --optimizer adam,lr=0.00001 --clip_grad_norm 5 --epoch_size 1000 --max_epoch 20 --accumulate_gradients 1 --lambda_qlm 1 --lambda_rr 1 --num_neg 1 --qlm_steps enfr,enes,ende,fres,frde,esde --max_pairs 100000 --master_port 12223

# srun --label python train.py --exp_name short_qlm_0.45 --fp16 true --amp 2 --model_type mbert --model_path bert-base-multilingual-uncased --batch_size 4 --qlm_mask_mode query --mlm_probability 0.45 --optimizer adam,lr=0.00001 --clip_grad_norm 5 --epoch_size 1000 --max_epoch 20 --accumulate_gradients 1 --lambda_qlm 1 --lambda_rr 1 --num_neg 1 --qlm_steps enfr,enes,ende,fres,frde,esde --max_pairs 100000 --master_port 12223

# srun --label python train.py --exp_name short_qlm_doc --fp16 true --amp 2 --model_type mbert --model_path bert-base-multilingual-uncased --batch_size 4 --qlm_mask_mode document --mlm_probability 0.30 --optimizer adam,lr=0.00001 --clip_grad_norm 5 --epoch_size 1000 --max_epoch 20 --accumulate_gradients 1 --lambda_qlm 1 --lambda_rr 1 --num_neg 1 --qlm_steps enfr,enes,ende,fres,frde,esde --max_pairs 100000 --master_port 12223

srun --label python train.py --exp_name short_qlm_mix --fp16 true --amp 2 --model_type mbert --model_path bert-base-multilingual-uncased --batch_size 4 --qlm_mask_mode mix --mlm_probability 0.30 --optimizer adam,lr=0.00001 --clip_grad_norm 5 --epoch_size 1000 --max_epoch 20 --accumulate_gradients 1 --lambda_qlm 1 --lambda_rr 1 --num_neg 1 --qlm_steps enfr,enes,ende,fres,frde,esde --max_pairs 100000 --master_port 12223