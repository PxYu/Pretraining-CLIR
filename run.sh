export CUDA_VISIBLE_DEVICES=1
python finetune-qa.py --model_type mbert --model_name bert-base-multilingual-uncased --batch_size 16 --num_epochs 10 --num_ft_layer 12 --squad_version 2.0

