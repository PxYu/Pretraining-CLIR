export NGPU=8;
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;
# need to calculate epoch_size (epoch_size's unit is number of positive qd pair) very carefully
# e.g., for each language pair with each task, # pos-qd-pairs / epoch = epoch_size * NGPU * batch_size
python -m torch.distributed.launch --nproc_per_node=$NGPU train.py --exp_name test \
                                                                --fp16 true \
                                                                --amp 2 \
                                                                --model_type mbert-long \
                                                                --model_path mBertLong/mBert-base-p1024-w32-0 \
                                                                --batch_size 4 \
                                                                --qlm_mask_mode query \
                                                                --mlm_probability 0.30 \
                                                                --optimizer adam,lr=0.00001 \
                                                                --clip_grad_norm 5 \
                                                                --epoch_size 1000 \
                                                                --max_epoch 10 \
                                                                --accumulate_gradients 1 \
                                                                --lambda_qlm 1 \
                                                                --lambda_rr 1 \
                                                                --num_neg 1 \
                                                                --max_pairs 100000 \
                                                                --rr_steps enfr,enes,ennl,esnl,fres,frnl \
								--slurm_debug true
