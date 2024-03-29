python -m torch.distributed.launch --nproc_per_node=10 finetune-search-5f.py --model_type $1 \
                                                                            --model_path $2 \
                                                                            --dataset $3 \
                                                                            --source_lang $4 \
                                                                            --target_lang $5 \
                                                                            --batch_size $6 \
                                                                            --full_doc_length \
                                                                            --num_neg 1 \
                                                                            --eval_step 1 \
                                                                            --num_epochs 20 \
                                                                            --apex_level O2 \
                                                                            --encoder_lr 2e-5 \
                                                                            --projector_lr 2e-5 \
                                                                            --num_ft_encoders 3 \
                                                                            --seed 611
                                                                            # --cased ####!!!!! be careful with this
